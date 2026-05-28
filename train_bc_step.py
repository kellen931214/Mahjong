import torch
import torch.nn.functional as F

def train_bc_step(model, batch, optimizer=None, lambdas=(0.6, 0.3, 0.1)):
    """
    BC 訓練/驗證步驟 - 計算多任務損失
    
    Args:
        model: DecisionMamba 模型
        batch: 批次數據 (已通過 bc_collate_fn 處理)
        optimizer: 優化器 (如果是 None，則代表處於 Validation 模式，不更新權重)
        lambdas: (action_lambda, rtg_lambda, state_lambda) 損失權重
    """
    # 根據是否傳入 optimizer 決定是否切換為 train 模式
    if optimizer is not None:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval() # 驗證模式
    
    device = next(model.parameters()).device
    
    rtg = batch["rtg"].to(device)               
    state = batch["state"].to(device)           
    target_action = batch["action"].to(device)  
    timesteps = batch["timesteps"].to(device)   
    
    batch_size, seq_len = target_action.shape
    start_token = torch.full((batch_size, 1), 180, dtype=torch.long, device=device)
    input_action = torch.cat([start_token, target_action[:, :-1]], dim=1)
    
    # 將 -100 (padding) 替換為 180 作為輸入給模型的佔位符
    input_action = torch.where(input_action < 0, torch.tensor(180, device=device), input_action)
    
    # 獲取模型的三個輸出
    pred_action, pred_rtg, pred_state, _ = model(
        rtg=rtg, 
        state=state, 
        action=input_action, 
        timesteps=timesteps
    )
    
    # ================= 1. 動作預測損失 (使用 ignore_index) =================
    action_dim = pred_action.shape[-1]
    
    ce_loss = F.cross_entropy(
        pred_action.reshape(-1, action_dim), 
        target_action.reshape(-1), 
        ignore_index=-100  # PyTorch 會自動幫我們略過 padding 的計算
    )
    
    # 計算 Valid Mask (後續的 RTG, State, Accuracy 都需要用到)
    valid_mask = (target_action != -100).float()
    
    # 計算 action accuracy
    pred_actions = pred_action.argmax(dim=-1)
    correct = ((pred_actions == target_action).float() * valid_mask).sum()
    accuracy = correct / (valid_mask.sum() + 1e-8)
    
    # ================= 2. RTG 預測損失 =================
    rtg_diff = (pred_rtg.squeeze(-1) - rtg.squeeze(-1)) ** 2
    rtg_loss = (rtg_diff * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    
    # ================= 3. 狀態預測損失 (Masked MSE) =================
    if pred_state is not None:
        # 分離空間特徵與純量特徵的掩碼
        spatial_mask = (state[:, :, :1360].abs() > 1e-6).float()
        scalar_mask = torch.ones_like(state[:, :, 1360:])
        state_mask = torch.cat([spatial_mask, scalar_mask], dim=-1)
        
        state_diff = (pred_state - state) ** 2
        state_loss = (state_diff * state_mask).sum() / (state_mask.sum() + 1e-8)
    else:
        state_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    
    # 三部分損失組合
    l1, l2, l3 = lambdas if len(lambdas) >= 3 else (0.6, 0.3, 0.1)
    total_loss = l1 * ce_loss + l2 * rtg_loss + l3 * state_loss
    
    # ================= 權重更新 (僅在訓練模式) =================
    if optimizer is not None:
        total_loss.backward()
        
        # ✨【修正 Bug】在 optimizer.step() 之前進行梯度裁剪，真正防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    
    return total_loss.item(), {
        "action_loss": ce_loss.item(), 
        "rtg_loss": rtg_loss.item(),
        "state_loss": state_loss.item(),
        "accuracy": accuracy.item()
    }