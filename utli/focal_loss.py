"""
utli/focal_loss.py — 多分類 Focal Loss 模組（MahjongFocalLoss）

專為日本麻將 AI 的長尾不平衡（Class Imbalance）場景設計。
核心公式：FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

參考文獻：
  - Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    "Focal Loss for Dense Object Detection." ICCV 2017.

適用於時序序列幾何形狀：(Batch_Size, Seq_Len, 181) → 內部自動展平。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MahjongFocalLoss(nn.Module):
    """
    多分類 Focal Loss，繼承 nn.Module。

    專為 (B, T, 181) 形狀的時序 Logits 設計，內部自動展平處理。
    透過非線性調製係數 (1 - p_t)^γ 自動打壓高頻易分類動作（如 Dahai）的梯度，
    並強行放大低頻難分類動作（如 Kong）的梯度。

    數學定義 ──────────────────────────────────────────────
      令 p_t = softmax(logits)[target_class]
      FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

      其中：
        - p_t        : 模型對真實類別的預測機率
        - (1 - p_t)^γ: 非線性調製係數（Modulating Factor）
        - α_t        : 類別平衡權重（Class-Balancing Weight）
        - γ          : 聚焦指數（Focusing Parameter）

    Gamma (γ) 的學術用途 ─────────────────────────────────
      γ 控制模型對「易分類樣本」的關注衰減速度：
        - γ = 0  → 退化成標準 CrossEntropyLoss
        - γ = 1  → 線性衰減，難易樣本的梯度差距以一次方區分
        - γ = 2  → 二次衰減，極度不平衡場景下效果最佳（文獻預設）

      範例（γ = 2.0）：
        若 p_t = 0.9（Dahai 易分類）：
          focal_weight = (1 - 0.9)^2 = 0.01
          → 梯度被壓縮 100×，模型不再浪費容量在已學會的切牌上
        若 p_t = 0.1（Kong 難分類）：
          focal_weight = (1 - 0.1)^2 = 0.81
          → 梯度僅衰減 19%，難分樣本的學習訊號被完整保留

    Alpha (α) 的學術用途 ─────────────────────────────────
      α_t 是每個類別的先驗平衡權重，用於補償不同類別在訓練集中
      的出現頻率差異。在此麻將場景中：
        - 切牌 (Dahai, 0~73)    : α = 1.0（基線）
        - 吃   (Chow,  74~103)  : α = 1.0
        - 碰   (Pong,  104~140) : α = 1.0
        - 槓   (Kong,  141~174) : α = 3.0（極稀有，需強行放大）
        - 特殊 (Special,175~180): α = 1.0
      即使 focal_weight 已經提供梯度調製，α 仍提供額外的
      類別級補償，兩者形成雙層加權機制。

    Args:
        alpha_weights : torch.Tensor, shape (181,), optional
            每個類別的先驗平衡權重。若為 None，則全部設為 1.0
            （即退化為僅含 γ 調製的純 Focal Loss）。
        gamma         : float, default=2.0
            聚焦指數 (Focusing Parameter)。值越大，對易分類樣本
            的壓制越強。文獻建議 0.0~5.0，不平衡場景通常用 2.0。
        ignore_index  : int, default=-100
            要忽略的目標索引（對齊 PyTorch CrossEntropyLoss 慣例）。
            用於跳過 padding token，使 loss 僅計算有效時間步。
        reduction     : str, default='mean'
            損失聚合方式：'mean'（預設）、'sum'、'none'。

    Input Shape:
        logits  : (Batch_Size, Seq_Len, 181) — 模型原始輸出（未經 softmax）
        targets : (Batch_Size, Seq_Len)       — 真實動作標籤，值域 [0, 180]

    Output Shape:
        loss    : scalar（reduction='mean'/'sum'）或 (B*T,)（reduction='none'）
    """

    def __init__(
        self,
        alpha_weights: torch.Tensor = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()

        # ── γ (Gamma)：聚焦指數，控制易分類樣本的梯度衰減速度 ──
        self.gamma = gamma

        # ── α (Alpha Weights)：類別平衡權重張量 ──
        # register_buffer 確保 alpha 跟隨模型一同遷移到 GPU，但不參與梯度計算
        if alpha_weights is not None:
            self.register_buffer("alpha", alpha_weights.clone().detach())
        else:
            self.register_buffer("alpha", torch.ones(181))

        # ── ignore_index：對齊 PyTorch CE loss 標準，跳過 padding token ──
        self.ignore_index = ignore_index

        # ── reduction：最終損失的聚合方式 ──
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction 必須為 'mean', 'sum', 或 'none'，收到: {reduction}"
            )
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        計算多分類 Focal Loss。

        Args:
            logits  : (B, T, 181) — 模型原始輸出 logits（未經 softmax）
            targets : (B, T)       — 真實動作標籤，padding 位置為 -100

        Returns:
            loss : scalar 或 (N,) tensor
        """
        # ══════════════════════════════════════════════════════════════
        # Step 0: 確保模組 buffer (alpha) 與輸入 tensor 在同一裝置上
        # ══════════════════════════════════════════════════════════════
        # register_buffer 不會自動跟隨外部 model.to(device)（因為此模組
        # 獨立於 DecisionMambaMultiHead 之外）。此處顯式同步，確保後續
        # alpha 索引操作不會觸發跨裝置錯誤。
        if self.alpha.device != logits.device:
            self.to(logits.device)

        # ══════════════════════════════════════════════════════════════
        # Step 1: 展平（Flatten）— 將時序維度合併到 Batch 維度
        # ══════════════════════════════════════════════════════════════
        # logits:  (B, T, 181) → (B*T, 181)
        # targets: (B, T)       → (B*T,)
        # 這樣做讓後續所有操作都基於二維張量，與標準分類 Loss 慣例相容
        num_classes = logits.shape[-1]
        logits_2d = logits.reshape(-1, num_classes)   # (B*T, 181)
        targets_1d = targets.reshape(-1)               # (B*T,)

        # ══════════════════════════════════════════════════════════════
        # Step 2: 建立 valid_mask（提早建立，供後續安全索引使用）
        # ══════════════════════════════════════════════════════════════
        # 在 bc_collate_fn 中，padding 位置被設為 -100。
        # 我們必須在 gather 之前就建立好 valid_mask，因為 gather 操作
        # 不接受負數索引（targets_1d 含 -100 會導致 RuntimeError）。
        valid_mask = (targets_1d != self.ignore_index)  # (B*T,), boolean

        # 若所有 token 都是 padding（防禦性保護），回傳 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        # ══════════════════════════════════════════════════════════════
        # Step 3: 安全化 targets — 將 ignore_index 替換為 0
        # ══════════════════════════════════════════════════════════════
        # gather 和 alpha 索引操作要求 targets 必須在 [0, num_classes-1]
        # 範圍內。我們將 padding 位置的 -100 替換為 0（任意合法索引），
        # 因為這些位置的 loss 在最終 reduction 時會被 valid_mask 排除。
        # 這是一個安全的技巧：padding 位置取出的 log_p_t / alpha_t 值
        # 雖然沒有意義，但最終不會被納入 loss 計算。
        targets_safe = targets_1d.clamp(min=0)  # (B*T,)，-100 → 0，其餘不變

        # ══════════════════════════════════════════════════════════════
        # Step 4: log_softmax — 數值穩定的對數機率正規化
        # ══════════════════════════════════════════════════════════════
        # 直接對 logits 進行 log_softmax，避免先 softmax 再取 log 的數值
        # 不穩定問題（防止 exp 溢位）。
        # log_p 形狀為 (B*T, 181)，每個元素為該類別的 log 機率。
        log_p = F.log_softmax(logits_2d, dim=-1)  # (B*T, 181)

        # ══════════════════════════════════════════════════════════════
        # Step 5: gather — 取出每個樣本對應真實類別的 log 機率
        # ══════════════════════════════════════════════════════════════
        # gather(dim, index) 沿著 dim 維度，用 index 指定的索引取值。
        # 使用 targets_safe（padding 位置已替換為 0）作為索引。
        # gather 結果 → (B*T, 1)，squeeze(1) → (B*T,)。
        # log_p_t[i] = log(softmax(logits_i)[target_i])
        log_p_t = log_p.gather(1, targets_safe.unsqueeze(1)).squeeze(1)  # (B*T,)

        # ══════════════════════════════════════════════════════════════
        # Step 6: p_t = exp(log_p_t) — 還原機率值
        # ══════════════════════════════════════════════════════════════
        # 由於 log_p_t 是 log 機率，取 exp 即為原始機率 p_t。
        # 此步驟是數值穩定的關鍵：我們從 log_softmax 出發，一路保持
        # log-space 運算，僅在需要計算 (1-p_t)^γ 時才還原到 linear-space。
        p_t = log_p_t.exp()  # (B*T,)，值域 [0, 1]

        # ══════════════════════════════════════════════════════════════
        # Step 7: focal_weight = (1 - p_t)^γ — 非線性調製係數
        # ══════════════════════════════════════════════════════════════
        # 這是 Focal Loss 的核心魔法：
        #   - 當 p_t → 1（模型很確定正確答案，如高頻 Dahai）：
        #     (1 - p_t) → 0，focal_weight → 0，loss 被大幅衰減
        #   - 當 p_t → 0（模型幾乎猜錯，如稀有 Kong）：
        #     (1 - p_t) → 1，focal_weight → 1，loss 完整保留
        # γ 指數決定了衰減曲線的陡峭程度：γ 越大，易分樣本被壓制得越快。
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
        focal_weight = (1.0 - p_t) ** self.gamma  # (B*T,)

        # ══════════════════════════════════════════════════════════════
        # Step 8: alpha_t — 類別級先驗加權
        # ══════════════════════════════════════════════════════════════
        # 從 alpha buffer 中按 targets_safe 索引取值。
        # alpha[141:175] = 3.0 會對所有 Kong 區間的樣本施加 3× 加權。
        # 這補償了 Kong 在訓練集中極低的出現頻率（可能 < 0.5%），
        # 確保模型不會完全忽略槓牌的梯度訊號。
        # 注意：padding 位置會被取到 alpha[0]=1.0，但最終被 mask 排除。
        alpha_t = self.alpha[targets_safe]  # (B*T,)

        # ══════════════════════════════════════════════════════════════
        # Step 9: Focal Loss 核心計算
        # ══════════════════════════════════════════════════════════════
        # FL = -α_t · (1 - p_t)^γ · log(p_t)
        #       ↑          ↑            ↑
        #   類別加權   調製係數   交叉熵核心
        #
        # 注意：此處沒有負號！因為 log_p_t 已經是 log(p_t)（負值），
        # 所以我們取 -log_p_t 才會得到正值 loss。實際上：
        #   loss_per_sample = α_t * focal_weight * (-log_p_t)
        # 這等價於公式中的 -α_t · (1-p_t)^γ · log(p_t)。
        loss_per_sample = alpha_t * focal_weight * (-log_p_t)  # (B*T,)

        # ══════════════════════════════════════════════════════════════
        # Step 10: reduction（聚合）— 只計算 valid_mask=True 的位置
        # ══════════════════════════════════════════════════════════════
        if self.reduction == "none":
            # 不回傳聚合後的標量，而是回傳每個有效樣本的逐點 loss
            # padding 位置設為 0（不影響後續自訂聚合）
            loss_per_sample = loss_per_sample * valid_mask.float()
            return loss_per_sample
        elif self.reduction == "sum":
            # 所有有效樣本的 loss 總和
            return loss_per_sample[valid_mask].sum()
        else:  # "mean"
            # 所有有效樣本的 loss 平均值（最常用的設定）
            return loss_per_sample[valid_mask].mean()


# ═══════════════════════════════════════════════════════════════════
# 輔助函數：快速建立預設的 alpha_weights 張量
# ═══════════════════════════════════════════════════════════════════

def build_default_alpha_weights(
    kong_weight: float = 3.0,
    chow_weight: float = 1.0,
    pong_weight: float = 1.0,
    dahai_weight: float = 1.0,
    special_weight: float = 1.0,
) -> torch.Tensor:
    """
    根據動作區間快速建立 alpha_weights 張量。

    動作區間（參照 evaluation_metrics.py 的 ACTION_BINS）：
      - dahai  : 0~73   (74 dims)  — 切牌 + 摸切
      - chow   : 74~103 (30 dims)  — 吃
      - pong   : 104~140(37 dims)  — 碰
      - kong   : 141~174(34 dims)  — 槓（最稀有）
      - special: 175~180(6 dims)   — 特殊動作

    Args:
        kong_weight   : 槓牌區間 (141~174) 的加權倍率，預設 3.0
        chow_weight   : 吃牌區間 (74~103)  的加權倍率，預設 1.0
        pong_weight   : 碰牌區間 (104~140) 的加權倍率，預設 1.0
        dahai_weight  : 切牌區間 (0~73)    的加權倍率，預設 1.0
        special_weight: 特殊區間 (175~180) 的加權倍率，預設 1.0

    Returns:
        alpha: torch.Tensor, shape (181,)
    """
    alpha = torch.ones(181)
    alpha[0:74] = dahai_weight       # Dahai
    alpha[74:104] = chow_weight      # Chow
    alpha[104:141] = pong_weight     # Pong
    alpha[141:175] = kong_weight     # Kong
    alpha[175:181] = special_weight  # Special
    return alpha