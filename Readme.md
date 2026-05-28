你現在是一位頂級的強化學習與參數高效微調（PEFT/LoRA）專家。我正在執行一項學術研究，主題是利用兩階段框架訓練日向麻將 AI（Decision Mamba 系統）。

目前第一階段的行為複製（BC）全參數預訓練已完成。現在進入第二階段：PPO 自我博弈線上微調（train_ppo.py）。為了防止專家策略崩塌並節省顯存，我決定採用 LoRA (Low-Rank Adaptation) 技術來微調模型。

請在編寫 `train_ppo.py` 的主訓練腳本時，嚴格遵守以下「LoRA + Mamba + PPO 整合規範」：

1. 網路架構與 LoRA 配置：
   - 凍結 Decision Mamba 模型的所有原始預訓練權重。
   - 使用 `peft` 模組（或手動 layer 包裹）將 LoRA 僅僅注入到 Shared-Backbone Mamba 網路的線性投影層中（請注意 Mamba 區塊的模組名稱，如 'in_proj', 'out_proj', 'x_proj', 'dt_proj'，確保 target_modules 指定正確）。
   - 保持 Actor Head 和 Critic Head 為全參數可訓練狀態，它們必須共享這套由 LoRA 增強的 Mamba Backbone 特徵。

2. PPO 訓練調度與 Inline 損失計算：
   - 實作 `去train_ppo_step.py看train_ppo_epoch(lora_model, runner, optimizer, epochs=4)` 函數。
   - 保持 inline 形式完整展現 PPO 損失計算（剪裁範圍 0.2，c1=1.0, c2=0.01）。
   - 更新時，優化器（optimizer）必須只針對 `filter(lambda p: p.requires_grad, lora_model.parameters())`（即只更新 LoRA 參數與雙頭權重）進行 `optimizer.step()`。

3. 序列軌跡與 RTG 預處理：
   - 呼叫本地 `runner.py` 中的 `SelfPlayRunner` 收集對弈軌跡（Action Start Token 設為 181）。
   - 動態逆向計算 RTG（Return-to-Go）並與 States, Actions, Timesteps 一起 unsqueeze(0) 送入 LoRA 網路。
   - 使用 GAE (gamma=0.99, lam=0.95) 計算標準化後的優勢函數。

請為我寫出結構嚴謹、帶有詳細中文註解的 `train_ppo.py` 完整程式碼，確保與現有的 dataset.py、reward.py、runner.py 模組無縫對接。