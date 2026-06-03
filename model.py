import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.rank = rank
        self.alpha = alpha
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        lora_out = (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)
        return self.linear(x) + lora_out

class MultiGrainedBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.conv1d_cg = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2)
        self.conv1d_fg = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2)
        
        self.proj_z_cg = LoRALinear(d_model, d_model)
        self.proj_z_fg = LoRALinear(d_model, d_model)
        
        self.ssm_cg = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.ssm_fg = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        
        self.fusion_norm = nn.LayerNorm(d_model)
        self.out_proj = LoRALinear(d_model, d_model)

    def forward(self, h_i_minus_1):
        h_i = self.norm(h_i_minus_1)
        
        h_cg_conv = self.conv1d_cg(h_i.transpose(1, 2))[:, :, :-2].transpose(1, 2)
        h_fg_conv = self.conv1d_fg(h_i.transpose(1, 2))[:, :, :-2].transpose(1, 2)
        
        h_cg_in = F.silu(h_cg_conv)
        h_fg_in = F.silu(h_fg_conv)
        
        z_cg = self.proj_z_cg(h_i_minus_1)
        z_fg = self.proj_z_fg(h_i_minus_1)
        
        h_cg = self.ssm_cg(h_cg_in)
        h_fg = self.ssm_fg(h_fg_in)
        
        h_cg_gated = h_cg * F.silu(z_cg)
        h_fg_gated = h_fg * F.silu(z_fg)
        
        h_mg = self.fusion_norm(h_cg_gated + h_fg_gated)
        
        return self.out_proj(h_mg) + h_i_minus_1

class DecisionMamba(nn.Module):
    def __init__(self, d_model=512, action_dim=181, state_dim=1380, max_ep_len=2048):
        super().__init__()
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_state = nn.Linear(state_dim, d_model)
        self.embed_action = nn.Embedding(action_dim, d_model)
        self.embed_timestep = nn.Embedding(max_ep_len, d_model) 
        self.input_proj = nn.Linear(3 * d_model, d_model)
        
        self.block = MultiGrainedBlock(d_model)
        
        self.head_action = nn.Linear(d_model, action_dim) 
        self.head_rtg = nn.Linear(d_model, 1)
        self.head_state = nn.Linear(d_model, state_dim)

    def prepare_for_ppo(self):
        """
        🔒 凍結 Backbone（embedding / input_proj / SSM / conv / LayerNorm），
        僅保留 LoRA 注入層（lora_A, lora_B）+ Actor/Critic/State Head 可訓練。
        
        這確保：
        1. LoRA 低秩適配真正發揮作用（不讓 backbone 被 PPO 拉扯偏離 BC 策略）
        2. 顯存使用極小化（>95% 參數凍結）
        3. 防止災難性遺忘
        """
        print("🔒 啟用 LoRA 模式：凍結 Backbone，僅訓練 LoRA 注入層 + Actor/Critic Head...")
        
        # 第一階段：全部凍結
        for param in self.parameters():
            param.requires_grad = False
        
        # 第二階段：選擇性解凍 — 僅 LoRA 參數 + 三個輸出 Head
        for name, param in self.named_parameters():
            # LoRA 低秩矩陣（存在於 MultiGrainedBlock 內的 LoRALinear）
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
            # 輸出 Head（需要從頭學習 PPO 的 policy/value/state 預測）
            elif name.startswith("head_action.") or name.startswith("head_rtg.") or name.startswith("head_state."):
                param.requires_grad = True
        
        # 統計
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"   可訓練: {trainable:,} / 凍結: {total - trainable:,} (比例: {trainable / total * 100:.2f}%)")

    def forward(self, rtg, state, action, timesteps=None):
        action_safe = torch.where(action < 0, torch.tensor(0, device=action.device, dtype=action.dtype), action)
        
        e_concat = torch.cat([self.embed_rtg(rtg), self.embed_state(state), self.embed_action(action_safe)], dim=-1)
        h0 = self.input_proj(e_concat)
        
        if timesteps is not None:
            h0 = h0 + self.embed_timestep(timesteps)
            
        h = self.block(h0)
        
        return self.head_action(h), self.head_rtg(h), self.head_state(h), h


class MahjongMultiHeadOutput(nn.Module):
    """
    多頭注意力分權輸出層 —— 評審團拼接機制（Jury Concatenation）

    將原本單一的 nn.Linear(512, 181) 替換為五個專職的線性預測頭，
    對同一個 Mamba 隱狀態（hidden state）同時進行並行計算，
    最後在 dim=-1 首尾拼接（torch.cat）還原為精確的 181 維向量。

    拆分依據（來自 mjx/include/internal/action.cpp Encode 函數）：
      - Discard Head : 0~73   (74 dims)  ─ DISCARD + TSUMOGIRI
      - Chow Head    : 74~103 (30 dims)  ─ CHI（含赤牌變體）
      - Pong Head    : 104~140(37 dims)  ─ PON（含赤牌變體）
      - Kong Head    : 141~174(34 dims)  ─ CLOSED/OPEN/ADDED KAN
      - Special Head : 175~180(6 dims)   ─ TSUMO/RON/RIICHI/NINE_TERMINALS/NO/DUMMY
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        # 五個專職 Linear Heads
        self.head_discard = nn.Linear(d_model, 74)   # 切牌
        self.head_chow    = nn.Linear(d_model, 30)   # 吃
        self.head_pong    = nn.Linear(d_model, 37)   # 碰
        self.head_kong    = nn.Linear(d_model, 34)   # 槓
        self.head_special = nn.Linear(d_model, 6)    # 特殊動作

    def forward(self, x):
        """
        Args:
            x: Mamba backbone 輸出的隱狀態，shape (B, T, d_model)

        Returns:
            logits: 拼接後的完整動作分數，shape (B, T, 181)
        """
        discard = self.head_discard(x)   # (B, T, 74)
        chow    = self.head_chow(x)      # (B, T, 30)
        pong    = self.head_pong(x)      # (B, T, 37)
        kong    = self.head_kong(x)      # (B, T, 34)
        special = self.head_special(x)   # (B, T, 6)

        return torch.cat([discard, chow, pong, kong, special], dim=-1)  # (B, T, 181)


class DecisionMambaMultiHead(DecisionMamba):
    """
    Decision Mamba 多頭分權版本

    繼承原有的 DecisionMamba 骨幹（embedding / MultiGrainedBlock / Head RTG / Head State），
    僅將 head_action 從單一 nn.Linear(512, 181) 替換為 MahjongMultiHeadOutput。

    forward() 介面完全相容，無需修改任何下游訓練/推論程式碼。
    """

    def __init__(self, d_model=512, action_dim=181, state_dim=1380, max_ep_len=2048):
        # 先呼叫父類別 __init__ 建立完整骨幹（含舊的 head_action 線性層）
        super().__init__(d_model=d_model, action_dim=action_dim,
                         state_dim=state_dim, max_ep_len=max_ep_len)
        # 覆蓋 head_action 為多頭輸出層
        self.head_action = MahjongMultiHeadOutput(d_model)
