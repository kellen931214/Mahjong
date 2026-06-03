"""
utli/lora_focal_finetune.py — LoRA 喚醒與 Focal Loss 微調初始化

實作「大腦骨幹凍結 ＋ 喚醒 LoRA 增量更新 ＋ Focal Loss 焦點衝刺」
兩階段微調策略的初始化邏輯。

核心功能：
  1. 精準凍結/解凍：遍歷 model.named_parameters()，鎖定 Backbone，
     僅解凍 lora_A / lora_B 低秩矩陣與 head_action.* 多頭輸出層。
  2. alpha_weights 建構：對 Kong 動作區間 (141~174) 施加 3.0× 基礎加權。
  3. Focal Loss 實例化：γ=2.0，搭配上述 alpha_weights。
  4. AdamW 過濾器：確保只有 requires_grad=True 的參數進入優化器。

使用範例：
    from model import DecisionMambaMultiHead
    from utli.lora_focal_finetune import setup_lora_focal_finetune

    model = DecisionMambaMultiHead()
    optimizer, focal_loss = setup_lora_focal_finetune(model, lr=3e-4)
    # 之後在 train_bc_step 中使用 loss_mode="focal", focal_loss=focal_loss
"""

import torch
import torch.nn as nn
import torch.optim as optim

from utli.focal_loss import MahjongFocalLoss, build_default_alpha_weights


def setup_lora_focal_finetune(
    model: nn.Module,
    lr: float = 3e-4,
    gamma: float = 2.0,
    kong_weight: float = 3.0,
    weight_decay: float = 1e-4,
    verbose: bool = True,
):
    """
    LoRA 喚醒與 Focal Loss 微調初始化。

    精確控制梯度流向，防止模型發生災難性遺忘（Catastrophic Forgetting）：

    Stage 1 ─ 全體凍結
        將 model.parameters() 中所有參數的 requires_grad 設為 False，
        確保沒有任何 Backbone 權重意外參與梯度更新。

    Stage 2 ─ 選擇性解凍
        遍歷 model.named_parameters()，精確解凍兩類參數：
          (a) 包含 "lora_A" 或 "lora_B" 的參數
              → 存在於 MultiGrainedBlock 內的所有 LoRALinear 注入層
              → 這些是唯一被允許更新的 Backbone 內部權重
          (b) name.startswith("head_action.") 的參數
              → MahjongMultiHeadOutput 的五個專職 Linear Head
              → 負責從凍結隱狀態中重新學習分類邊界

        注意：head_rtg.* 與 head_state.* 不被解凍（與 prepare_for_ppo 不同），
        因為此階段僅專注於提升動作分類的 Kong 準確率。

    Stage 3 ─ Focal Loss 建構
        使用 build_default_alpha_weights(kong_weight=3.0) 建立 (181,) 的
        alpha_weights，其中 Kong 區間 (141~174) 設為 3.0×，其餘設為 1.0×。
        以 γ=2.0 實例化 MahjongFocalLoss。

    Stage 4 ─ AdamW 過濾器
        使用 filter(lambda p: p.requires_grad, model.parameters()) 精確過濾，
        確保只有 LoRA 參數與多頭參數被送入 optim.AdamW。
        學習率設為 3e-4（微調專用，比預訓練 5e-4 稍保守）。

    Args:
        model        : nn.Module
            DecisionMambaMultiHead 模型實例。
        lr           : float, default=3e-4
            LoRA 微調學習率。比預訓練 (5e-4) 略保守，防止 LoRA 矩陣劇烈震盪。
        gamma        : float, default=2.0
            Focal Loss 聚焦指數。2.0 對極度不平衡場景效果最佳。
        kong_weight  : float, default=3.0
            Kong 動作區間 (141~174) 的 alpha 加權倍率。
        weight_decay : float, default=1e-4
            AdamW 權重衰減係數。
        verbose      : bool, default=True
            是否印出詳細的凍結/解凍統計資訊。

    Returns:
        optimizer   : torch.optim.AdamW
            僅包含 requires_grad=True 參數的優化器。
        focal_loss  : MahjongFocalLoss
            已配置 alpha_weights 與 gamma 的 Focal Loss 實例。
    """
    # ══════════════════════════════════════════════════════════════════
    # Stage 1: 全體凍結（Freeze All）
    # ══════════════════════════════════════════════════════════════════
    # 將所有參數的 requires_grad 設為 False。
    # 這一步確保沒有任何 Backbone 權重（Mamba SSM、Conv1d、LayerNorm、
    # Embedding、Input Proj 等）會在後續微調中被意外更新。
    # 這是防止災難性遺忘的第一道防線。
    if verbose:
        print("\n" + "=" * 70)
        print("🔒 LoRA 喚醒微調初始化：三階段精準梯度控制")
        print("=" * 70)
        print("Stage 1/4: 全體凍結（Freeze All）...")

    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  已凍結全部 {total_params:,} 個參數")

    # ══════════════════════════════════════════════════════════════════
    # Stage 2: 選擇性解凍（Selective Unfreeze）
    # ══════════════════════════════════════════════════════════════════
    # 遍歷所有具名參數，精確解凍兩類關鍵參數：
    #
    #  (a) lora_A / lora_B — 低秩適配矩陣
    #      這些參數存在於 MultiGrainedBlock 內每個 LoRALinear 中：
    #        - proj_z_cg.lora_A, proj_z_cg.lora_B
    #        - proj_z_fg.lora_A, proj_z_fg.lora_B
    #        - out_proj.lora_A, out_proj.lora_B
    #      LoRA 的理念是：凍結預訓練權重 W，只訓練低秩增量 ΔW = A·B。
    #      這樣做既保留原始知識，又允許模型適應新數據分佈。
    #
    #  (b) head_action.* — 多頭輸出層
    #      MahjongMultiHeadOutput 內的五個 Linear Head：
    #        - head_action.head_discard (512→74)
    #        - head_action.head_chow    (512→30)
    #        - head_action.head_pong    (512→37)
    #        - head_action.head_kong    (512→34)
    #        - head_action.head_special (512→6)
    #      這些 Head 需要從頭學習將凍結隱狀態映射到正確的動作分類邊界，
    #      尤其是 Kong Head 需要在 Focal Loss 的輔助下學會稀有槓牌。
    #
    # 注意：不解凍 head_rtg.* 與 head_state.*，因為此階段目標是
    #       提升動作分類準確率，而非改善 RTG/State 預測。
    if verbose:
        print("\nStage 2/4: 選擇性解凍（Selective Unfreeze）...")

    trainable_count = 0
    lora_count = 0
    head_action_count = 0

    for name, param in model.named_parameters():
        should_unfreeze = False
        reason = ""

        # 檢查是否為 LoRA 低秩矩陣參數
        if "lora_A" in name or "lora_B" in name:
            should_unfreeze = True
            reason = "LoRA"
            lora_count += param.numel()

        # 檢查是否為多頭輸出層參數
        elif name.startswith("head_action."):
            should_unfreeze = True
            reason = "MultiHead Action"
            head_action_count += param.numel()

        if should_unfreeze:
            param.requires_grad = True
            trainable_count += param.numel()
            if verbose:
                print(f"  ✅ 解凍 [{reason:>16s}] {name:<60s} shape={list(param.shape)}")

    frozen_count = total_params - trainable_count

    if verbose:
        print(f"\n  📊 凍結統計：")
        print(f"     總參數:         {total_params:>12,}")
        print(f"     凍結 (Backbone): {frozen_count:>12,}  ({frozen_count/total_params*100:.2f}%)")
        print(f"       └ 其中 LoRA:   {lora_count:>12,}  ({lora_count/total_params*100:.2f}%)")
        print(f"       └ 其中 Head:   {head_action_count:>12,}  ({head_action_count/total_params*100:.2f}%)")
        print(f"     可訓練:          {trainable_count:>12,}  ({trainable_count/total_params*100:.2f}%)")

    # ══════════════════════════════════════════════════════════════════
    # Stage 3: Focal Loss 建構
    # ══════════════════════════════════════════════════════════════════
    # 建立 alpha_weights：對 Kong (141~174) 施加 kong_weight 倍加權。
    # 以 γ=gamma（預設 2.0）實例化 MahjongFocalLoss。
    # 此 Loss 會自動打壓 Dahai 的梯度，放大 Kong 的梯度。
    if verbose:
        print(f"\nStage 3/4: 建構 Focal Loss（γ={gamma}, kong_weight={kong_weight}×）...")

    alpha = build_default_alpha_weights(kong_weight=kong_weight)
    focal_loss = MahjongFocalLoss(alpha_weights=alpha, gamma=gamma)

    if verbose:
        print(f"  alpha_weights 區間分佈：")
        print(f"    Dahai  (0~73)   : {alpha[0:74].unique().tolist()}")
        print(f"    Chow   (74~103) : {alpha[74:104].unique().tolist()}")
        print(f"    Pong   (104~140): {alpha[104:141].unique().tolist()}")
        print(f"    Kong   (141~174): {alpha[141:175].unique().tolist()}  ← 加權目標")
        print(f"    Special(175~180): {alpha[175:181].unique().tolist()}")
        print(f"  MahjongFocalLoss 實例化完成 (γ={gamma}, ignore_index=-100, reduction='mean')")

    # ══════════════════════════════════════════════════════════════════
    # Stage 4: AdamW 過濾器
    # ══════════════════════════════════════════════════════════════════
    # 使用 filter() 精確過濾，確保只有 requires_grad=True 的參數
    # 會被送進 AdamW 優化器。這是防止 Backbone 被意外更新的
    # 第二道（也是最後一道）防線。
    #
    # 學習率設為 lr=3e-4：
    #   - 比預訓練階段 (5e-4) 略低，因為 LoRA 矩陣通常對學習率敏感
    #   - LoRA 文獻（Hu et al., 2021）建議微調學習率在 1e-4~5e-4 之間
    #   - 3e-4 在保守與效率之間取得平衡
    if verbose:
        print(f"\nStage 4/4: 設定 AdamW 過濾器（lr={lr}, weight_decay={weight_decay}）...")

    # filter() 回傳的是一個迭代器，AdamW 會內部將其轉為 param_groups
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # 驗證：優化器管理的參數數量必須等於 requires_grad=True 的參數數量
    opt_param_count = sum(
        p.numel() for group in optimizer.param_groups for p in group["params"]
    )

    if verbose:
        print(f"  優化器管理參數: {opt_param_count:,}")
        assert opt_param_count == trainable_count, (
            f"❌ 優化器參數數量 ({opt_param_count}) 與可訓練參數 ({trainable_count}) 不一致！"
        )
        print(f"  ✅ 過濾驗證通過：AdamW 僅包含 LoRA + MultiHead 參數")
        print("=" * 70 + "\n")

    return optimizer, focal_loss


# ═══════════════════════════════════════════════════════════════════
# 便利函數：一行完成初始化並回傳所有必要元件
# ═══════════════════════════════════════════════════════════════════

def create_finetune_components(
    model: nn.Module,
    lr: float = 3e-4,
    gamma: float = 2.0,
    kong_weight: float = 3.0,
    weight_decay: float = 1e-4,
) -> dict:
    """
    一行完成 LoRA 喚醒微調的所有元件初始化。

    這是 setup_lora_focal_finetune 的便利包裝，回傳一個 dict，
    方便直接解包傳入訓練迴圈。

    Args:
        model        : DecisionMambaMultiHead 模型實例
        lr           : LoRA 微調學習率 (default: 3e-4)
        gamma        : Focal Loss 聚焦指數 (default: 2.0)
        kong_weight  : Kong 區間 alpha 加權 (default: 3.0)
        weight_decay : AdamW 權重衰減 (default: 1e-4)

    Returns:
        dict: {
            "optimizer":  AdamW optimizer (僅含可訓練參數),
            "focal_loss": MahjongFocalLoss 實例,
            "loss_mode":  "focal" (固定值，方便傳入 train_bc_step),
        }

    Usage:
        model = DecisionMambaMultiHead()
        components = create_finetune_components(model, lr=3e-4)

        for batch in train_loader:
            loss, metrics = train_bc_step(
                model, batch,
                optimizer=components["optimizer"],
                loss_mode=components["loss_mode"],
                focal_loss=components["focal_loss"],
            )
    """
    optimizer, focal_loss = setup_lora_focal_finetune(
        model=model,
        lr=lr,
        gamma=gamma,
        kong_weight=kong_weight,
        weight_decay=weight_decay,
    )
    return {
        "optimizer": optimizer,
        "focal_loss": focal_loss,
        "loss_mode": "focal",
    }