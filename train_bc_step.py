import torch
import torch.nn.functional as F

def train_bc_step(model, batch, optimizer=None, lambdas=(0.6, 0.3, 0.1)):
    """
    BC 訓練/驗證步驟 - 計算多任務損失（嚴格因果對齊版）
    """
    if optimizer is not None:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()
    
    device = next(model.parameters()).device
    
    rtg = batch["rtg"].to(device)               
    state = batch["state"].to(device)           
    target_action = batch["target_action"].to(device)  
    input_action = batch["input_action"].to(device)   # 已在 bc_collate_fn 完成自迴歸右移
    timesteps = batch["timesteps"].to(device)   
    


    pred_action, pred_rtg, pred_state, _ = model(
        rtg=rtg, 
        state=state, 
        action=input_action, 
        timesteps=timesteps
    )
    
    # ================= 1. 動作預測損失 =================
    action_dim = pred_action.shape[-1]
    
    ce_loss = F.cross_entropy(
        pred_action.reshape(-1, action_dim), 
        target_action.reshape(-1), 
        ignore_index=-100
    )
    
    valid_mask = (target_action != -100).float()
    
    # 計算 action accuracy（排除 padding -100 與 none/跳過 179，後者本質是「非我回合」的空動作）
    eval_mask = valid_mask * (target_action != 179).float()
    pred_actions = pred_action.argmax(dim=-1)
    correct = ((pred_actions == target_action).float() * eval_mask).sum()
    accuracy = correct / (eval_mask.sum() + 1e-8)
    
    # ================= 2. RTG 預測損失 =================
    rtg_diff = (pred_rtg.squeeze(-1) - rtg.squeeze(-1)) ** 2
    rtg_loss = (rtg_diff * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    
    # ================= 3. 狀態預測損失 =================
    if pred_state is not None:
        spatial_mask = (state[:, :, :1360].abs() > 1e-6).float()
        scalar_mask = torch.ones_like(state[:, :, 1360:])
        state_mask = torch.cat([spatial_mask, scalar_mask], dim=-1)
        
        state_diff = (pred_state - state) ** 2
        state_loss = (state_diff * state_mask).sum() / (state_mask.sum() + 1e-8)
    else:
        state_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    
    l1, l2, l3 = lambdas if len(lambdas) >= 3 else (0.6, 0.3, 0.1)
    total_loss = l1 * ce_loss + l2 * rtg_loss + l3 * state_loss
    
    if optimizer is not None:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return total_loss.item(), {
        "action_loss": ce_loss.item(), 
        "rtg_loss": rtg_loss.item(),
        "state_loss": state_loss.item(),
        "accuracy": accuracy.item()
    }