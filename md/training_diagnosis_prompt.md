# 麻將 PPO + Decision Mamba 訓練診斷與修復 Prompt

你是一位熟悉 **Deep Reinforcement Learning、PPO、GAE、Decision Transformer / Decision Mamba、RTG-conditioned policy、LoRA fine-tuning、麻將 AI reward design** 的研究型工程師。

請你幫我完整診斷我目前的麻將 PPO 訓練問題，並且根據我提供的程式碼結構，指出可能的 bug、設計問題、應該改的檔案、應該新增的 log，以及最合理的實驗順序。

請不要只給概念解釋。
我要你用「可以實際改程式」的角度回答。

---

# 一、目前系統架構

## 模型

* 模型：Decision Mamba
* 狀態維度：1380
* 動作空間：181
* hidden dim：512
* 架構特性：RTG-conditioned SSM
* 每一步輸入包含：

  * return-to-go
  * state
  * action
  * timestep

## 訓練方式

我目前是從 BC 權重開始訓練：

1. 先用 Behavioral Cloning 預訓練 Decision Mamba。
2. 再使用 PPO + LoRA 做線上微調。
3. LoRA 階段只訓練：

   * LoRA 注入層
   * Actor Head
   * Critic Head

可訓練參數量：

```text
826K / 9.2M
```

## PPO config

```text
learning rate = 5e-5
clip epsilon = 0.2
gamma = 0.99
GAE lambda = 0.95
PPO epoch = 1
temperature = 0.8
```

---

# 二、目前遇到的核心問題

## 原始現象

Avg Reward 一直震盪，沒有明顯上升趨勢。

但其他指標又看起來有改善：

| 指標                  | 現象                      |
| ------------------- | ----------------------- |
| Value Loss          | 從 3~5 穩定下降到 0.03~0.04   |
| Policy Loss         | 約 0.06~0.10，穩定          |
| Avg Reward sparse   | 在 -0.5 ~ +0.6 間震盪，沒有趨勢  |
| 平均排名 window 50×32 局 | defense 從 2.76 改善到 2.10 |
| 贏牌率 window 50×32 局  | defense 從 18% 上升到 40%   |

核心矛盾是：

```text
Value Loss 幾乎完美收斂，
排名和贏牌率看起來也有改善，
但 Avg Reward 完全沒有穩定上升。
```

請你判斷這是否正常，以及這代表什麼。

---

# 三、我已經做過的修復

## 修復 1：Reward Ratio 調整

檔案：

```text
runner.py
```

原本 shaping reward 幾乎佔整局 reward 的 99%。

原本大概是：

```text
shaping reward 總量約 105
terminal reward 胡牌約 +0.5
terminal penalty 放銃約 -1.5
```

所以終局訊號只佔約 1%。

我後來改成：

```text
每步 shaping reward ÷ 30
terminal penalty × 15
terminal backward reward × 15
```

目標是讓 terminal reward 比例變高。

---

## 修復 2：Return Z-Score Normalization

檔案：

```text
train_ppo_step.py
```

舊版 value loss：

```python
value_loss = MSE / ret_std
```

問題是每個 batch 的 ret_std 不同，造成 value loss 尺度跳動。

新版改成：

```python
returns_norm = (returns - mean) / std
value_loss = MSE(values, returns_norm)
```

讓 Critic 預測目標固定在大概：

```text
[-3, +3]
```

---

## 修復 3：Avg Reward 計算修正

檔案：

```text
train_ppo_step.py
```

舊版錯誤：

```python
total_reward += rewards[-1]
```

只看最後一步 reward，導致 avg_reward 幾乎是噪音。

新版：

```python
total_reward += sum(rewards)
```

改成看整局 reward 總和。

---

## 修復 4：Window 統計修正

檔案：

```text
train_ppo_step.py
train_ppo.py
```

舊版錯誤：

```text
每 iter 32 局只取最後一局結果餵進 window
```

新版：

```text
每 iter 的 32 局結果全部餵進 window
```

---

# 四、修復後的結果

## Sparse mode

PPO vs BC，batch = 32 局：

```text
iter 1: -0.38
iter 2: -0.25
iter 3: -0.50
iter 4: 0.00
iter 5: 0.62
iter 6: -0.25
```

Avg Reward 還是震盪，沒有穩定上升。

---

## Dense mode

新版 reward，batch = 32 局：

```text
iter 1: 3.93
iter 2: 5.46
iter 3: 3.56
iter 4: 5.07
iter 5: 13.89
```

dense reward 數值比較大，因為混合了：

```text
shaping reward + terminal reward + backward reward
```

---

# 五、我的推測

請你逐一判斷以下推測是否成立，並說明原因。

---

## 推測 A：Dense Reward 可能被 Reward Hacking

我懷疑 shaping reward 還是太強。

目前 shaping 包含：

```text
向聽潛力
寶牌
向聽進展
```

即使我做了：

```text
shaping ÷ 30
terminal × 15
```

但模型可能仍然學到：

```text
最大化向聽數改善 / 牌型潛力
```

而不是：

```text
真正提高和牌率、降低放銃率、提高排名
```

已觀察到的現象：

```text
Sparse self-play 和了率 3.49% 很低
流局率 87% 很高
但一位率 26.2% 略高於隨機 25%
```

這可能代表 agent 學到極端防守策略：

```text
寧可流局，也不要放銃。
```

請你判斷這是不是 reward hacking，並告訴我要怎麼驗證。

---

## 推測 B：Self-Play 中 Avg Reward 本來就不會穩定上升

很多舊實驗是：

```text
PPO vs PPO
```

對手也會跟著變強，所以 reward 不一定會上升。

新版改成：

```text
PPO vs frozen BC
```

但只跑約 500 iter 就被 kill，所以數據量可能不夠。

請你判斷：

1. Self-play 是否不適合用 avg_reward 觀察訓練進展？
2. 固定 BC 對手是否比較適合看 agent 是否真的變強？
3. 應該怎麼設計 evaluation protocol？

---

## 推測 C：32 局不足以消除麻將隨機性

麻將 outcome 方差很大。

每 iter 只有：

```text
32 局 × 約 300 步
```

我懷疑這不足以讓 avg_reward 顯示穩定趨勢。

請你判斷：

1. 32 局是否太少？
2. sparse reward 下 avg_reward 震盪是否正常？
3. 至少需要 128、512、1000 還是多少局才能比較可信？
4. training rollout 和 evaluation games 應該分開設計嗎？

---

## 推測 D：RTG-conditioned 架構與 PPO 可能有 mismatch

Decision Mamba 是 RTG-conditioned。

每一步輸入含：

```text
return-to-go
state
action
timestep
```

我擔心 PPO rollout 和 train update 時的 RTG 不一致。

例如：

```text
rollout 時 action 是根據某個 rtg_input 取樣
但 PPO update 時重新 forward model 時可能用另一個 RTG
```

這會不會導致：

```python
ratio = exp(new_logprob - old_logprob)
```

不再是同一個 observation/context 下的 policy ratio？

請你幫我確認：

1. RTG input 是否應該被視為 observation/context 的一部分？
2. PPO update 時是否必須使用 rollout 當下存下來的 rtg_input？
3. rollout 時到底應該用真實 future return-to-go，還是 target_return - accumulated_reward？
4. Critic 預測 value 和 RTG conditioning 之間是否會產生資訊洩漏或語意混淆？
5. 這是否可能解釋 Value Loss 很漂亮但 Avg Reward 不上升？

---

## 推測 E：avg_reward 這個指標本身可能沒有意義

在 dense mode 下，avg_reward 是混合值：

```text
shaping accumulation + terminal outcome + backward reward
```

這可能無法反映真實麻將表現。

真正應該看的可能是：

```text
平均排名
一位率
二位率
top2 rate
和牌率
放銃率
流局率
平均點差
```

請你判斷：

1. avg_reward 是否應該只當 debug 指標？
2. dense avg_reward 是否不該作為主要 training progress metric？
3. 應該建立哪些更可靠的 evaluation metrics？

---

# 六、目前程式碼結構

```text
utli/
  rewards.py              ← MahjongRewardCalculator，計算 shaping + penalty + backward reward
  runner.py               ← SelfPlayRunner，負責遊戲對弈、軌跡收集、reward 組裝
  train_ppo_step.py       ← train_ppo_epoch，負責 batch 化、GAE、PPO update
  evaluation_metrics.py   ← 評估指標計算

train_ppo.py              ← 主訓練迴圈，載入 BC、加入 LoRA、PPO loop
evaluate.py               ← 離線評估 + self-play 評估入口
```

目前 reward 計算鏈大概是：

```text
runner.py:
  每步 shaping = potential + dora + progression
  shaping reward ÷ 30

  終局 penalty = reward_calculator × 15
  終局 backward = reward_calculator × 15，逐步回溯

train_ppo_step.py:
  sum(rewards) 整局累加
  avg_reward = total_reward / batch_size
```

---

# 七、我希望你幫我完成的任務

請你根據以上資訊，完成以下診斷。

---

## 任務 1：判斷目前現象是否合理

請回答：

```text
Value Loss 收斂
Policy Loss 穩定
排名與和率改善
但 Avg Reward 不上升
```

這種情況是否合理？

請分別從以下角度解釋：

1. PPO / GAE 的角度
2. Critic value loss 的角度
3. reward 設計的角度
4. 麻將高方差環境的角度
5. RTG-conditioned model 的角度

---

## 任務 2：診斷 reward design 問題

請幫我判斷目前 reward 是否可能有以下問題：

1. shaping reward 太強
2. terminal reward 太弱
3. backward reward 不該手動加
4. potential reward 不是 potential-based shaping
5. dense reward 導致 reward hacking
6. avg_reward 被 shaping 汙染

請你告訴我：

```text
應該保留哪些 reward
應該移除哪些 reward
應該怎麼縮放 shaping
terminal reward 應該放在哪一步
是否應該關掉 backward reward
```

請特別說明：

```text
為什麼 PPO / GAE 本身已經會把 terminal reward 往前傳，
所以不一定需要手動 backward reward。
```

---

## 任務 3：設計正確的 reward ablation 實驗

請幫我設計至少三組實驗：

```text
A. terminal only
B. terminal + small potential-based shaping
C. terminal + current dense shaping
```

每組請說明：

1. reward 怎麼設計
2. 要看哪些 metric
3. 預期現象是什麼
4. 如果出現某種結果，代表什麼問題

---

## 任務 4：設計 evaluation protocol

請幫我設計一套可靠的 evaluation protocol。

請包含：

1. training rollout 每 iter 要幾局
2. quick eval 要幾局
3. formal eval 要幾局
4. 每幾個 iter 評估一次
5. 固定對手怎麼設計
6. self-play opponent pool 怎麼設計
7. 如何比較 current PPO vs frozen BC
8. 如何比較 current PPO vs old checkpoint
9. 如何避免只用 32 局 avg_reward 誤判

請你用表格整理。

---

## 任務 5：檢查 RTG-conditioned PPO 的潛在問題

請詳細說明 Decision Mamba / Decision Transformer 類架構接 PPO 時要注意什麼。

請特別回答：

1. PPO ratio 的 new_logprob 和 old_logprob 是否必須使用完全相同的 RTG input？
2. rollout 時應該存下哪些欄位？
3. train update 時哪些欄位不能重新計算？
4. RTG 應該怎麼定義？
5. Critic input 是否應該包含 RTG？
6. Critic target 是否應該是 normalized return？
7. 如果 RTG 是真實 future return，是否可能造成 leakage？
8. 如果 rollout RTG 和 training RTG 不一致，會造成什麼後果？

請給我一份 code-level checklist。

---

## 任務 6：建議要新增的 logging

請列出我應該新增的 log。

至少包含：

```text
avg_total_reward
avg_terminal_reward
avg_shaping_reward
avg_backward_reward
episode_length
mean_rank
top1_rate
top2_rate
win_rate
deal_in_rate
draw_rate
avg_point_delta
entropy
approx_kl
clip_fraction
ratio_mean
ratio_min
ratio_max
adv_mean
adv_std
return_mean
return_std
value_loss
policy_loss
explained_variance
```

請你說明每個 log 用來判斷什麼問題。

---

## 任務 7：指出應該改的檔案

請根據我的程式碼結構，告訴我每個檔案應該檢查或修改什麼：

```text
utli/rewards.py
utli/runner.py
utli/train_ppo_step.py
utli/evaluation_metrics.py
train_ppo.py
evaluate.py
```

請用以下格式回答：

```text
檔案：
應檢查：
應修改：
新增 log：
可能 bug：
```

---

## 任務 8：給我一個最推薦的下一步修復順序

請不要一次叫我全改。

請給我一個最合理的順序，例如：

```text
Step 1：先關掉 backward reward
Step 2：跑 terminal-only PPO vs frozen BC
Step 3：新增 reward channel logging
Step 4：固定 eval protocol
Step 5：檢查 RTG input 是否 rollout/train 一致
Step 6：再加 potential-based shaping
Step 7：最後才調整 reward scale
```

每一步請說明：

1. 為什麼先做這一步
2. 要改哪裡
3. 預期會看到什麼
4. 如果結果不好，下一步怎麼判斷

---

# 八、我希望你最後給我的輸出格式

請用以下格式回答：

---

## 1. 總體判斷

簡短說明：

```text
目前這個現象是否正常？
最大的問題可能在哪裡？
最不應該相信哪個指標？
最應該相信哪些指標？
```

---

## 2. 逐一判斷我的五個推測

請用表格：

```text
推測 | 是否成立 | 原因 | 如何驗證
```

---

## 3. Reward Design 修復建議

請明確回答：

```text
terminal reward 要不要放大？
shaping 要不要保留？
backward reward 要不要關掉？
potential-based shaping 怎麼做？
avg_reward 要怎麼拆？
```

---

## 4. RTG-conditioned PPO 檢查清單

請列出我必須確認的 code-level checklist。

---

## 5. Evaluation Protocol

請給我一份推薦表格：

```text
用途 | 對手 | 局數 | 頻率 | 看哪些 metric
```

---

## 6. Logging 清單

請列出每個 log 的用途。

---

## 7. 各檔案修改建議

請逐檔說明：

```text
utli/rewards.py
utli/runner.py
utli/train_ppo_step.py
utli/evaluation_metrics.py
train_ppo.py
evaluate.py
```

---

## 8. 最推薦的下一步實驗順序

請給我一個可以照做的順序，不要只講理論。

---

# 九、重要限制

請注意：

1. 不要只說「多跑一點」。
2. 不要只說「reward 要調」。
3. 不要只說「看 avg reward」。
4. 請區分：

   * training reward
   * evaluation reward
   * terminal reward
   * shaping reward
   * backward reward
5. 請區分：

   * PPO rollout 用的 RTG
   * PPO update 用的 RTG
   * Critic 預測的 value
   * GAE 計算用的 return
6. 請用可以實際 debug 程式的方式回答。
7. 如果你認為某個設計是錯的，請直接指出。
8. 如果你認為某個指標沒有意義，請直接指出。
9. 如果你認為需要 ablation，請明確說要怎麼跑。
10. 如果你認為要新增 log，請明確說 log 在哪個檔案加。

請開始完整診斷。