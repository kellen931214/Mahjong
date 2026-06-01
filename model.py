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