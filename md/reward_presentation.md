# 🧠 Mahjong Reward Shaping 講稿 —— 條件式雙模式攻守獎勵系統

> 報告用講稿，預計報告時間：5–7 分鐘

---

## 投影片 1：為什麼需要 Reward Shaping？

各位好，今天要介紹我們在麻將 AI 訓練中設計的 Reward Shaping 系統。

首先我們要面對一個根本問題：**麻將是一個極度稀疏獎勵（Sparse Reward）的環境**。

什麼意思呢？在一般的 RL 設定裡，你打完整個半莊——也就是 8 到 12 局牌——才知道最終輸贏。中間的每一步，你都**沒有任何 reward signal**。這對 Critic（Value Function）的學習是災難性的：Critic 沒辦法穩定評估每一步狀態的好壞，Policy Gradient 的 variance 會極大，訓練要嘛不收斂，要嘛收斂到次優解。

還有第二個問題：麻將跟圍棋、西洋棋不同，它不是單純的 zero-sum。**麻將有「攻守抉擇」**——你手牌很爛的時候，硬要做大牌只會放銃送分；手牌很好的時候，保守棄和又會浪費机会。但傳統的 RL reward（終局輸贏）**完全無法傳達這個 nuance**。

所以我們設計了一套 **條件式雙模式攻守獎勵系統（Conditional Dual-Mode Reward Calculator）**，把稀疏的終局訊號，轉換為每一步都有的稠密梯度，讓模型能快速學會什麼時候該進攻、什麼時候該防守。

---

## 投影片 2：核心設計理念 —— 雙模式分離

我們的設計核心哲學很簡單，但很有效：

> **不中途換 reward function，而是直接訓練兩個獨立的模型。**

為什麼不中途換？因為 PPO 的 Critic 仰賴 reward function 是**靜態的（stationary）**。如果在同一局中，前面用進攻權重、後面忽然切防守權重，Critic 會看到同一個 state 對應不同的 return target，這叫 non-stationary reward——會讓 Value Loss 劇烈震盪，根本學不起來。

所以我們的解法是：

```
BC checkpoint（1380 dims 基礎特徵）
         │
    ┌────┴────┐
    ▼         ▼
Attack Model   Defense Model
(potential=0.4)  (potential=0.01)
(penalty=1.8)    (penalty=6.0)
    │         │
    └────┬────┘
         ▼
   各自 PPO 訓練
   各自存 checkpoint
```

訓練時透過 `--mode attack` 或 `--mode defense` 指定，`SelfPlayRunner` 初始化時就呼叫 `reward_calculator.set_mode(train_mode)`，**整場比賽全程固定模式**，保證 Critic 平穩學習。

---

## 投影片 3：五大 Reward 元件總覽

我們的 reward 由 **五個稠密元件** 組合而成。以下是總覽：

| # | 元件 | 型態 | 觸發時機 | 語義 |
|---|------|------|---------|------|
| 1 | `r_potential` | Mode-Aware | 每步 | 向聽潛力——「這手牌離聽牌多近？」 |
| 2 | `r_dora` | Mode-Aware | 每步 | 寶牌留存——「手上有幾張加分牌？」 |
| 3 | `r_progression` | Mode-Aware | 每步 | 向聽進展——「這一步改善了多少？」 |
| 4 | `r_penalty` | Mode-Aware | 終局觸發 | 放銃懲罰——「打出去被胡了要重罰」 |
| 5 | `r_backward` | Mode-Invariant | 終局觸發 | 回溯得分——「胡牌後回頭分配功勞」 |

前三個是**即時獎勵（Immediate Reward）**，每步都計算，讓 Critic 有稠密的 learning signal。
後兩個是**終局結算獎勵（Terminal Reward）**，只在終局觸發，負責 credit assignment。

---

## 投影片 4：① r_potential —— 向聽潛力獎勵（Mode-Aware）

這是五項 reward 中最核心的元件。

公式如下：
```
if shanten <= 0（已聽牌）:
    r_potential = base_exp_score / score_norm_factor
else:
    ukeire = 有效進張數（扣除場上已見的牌）
    r_potential = base_exp_score × ukeire / (shanten + 1) / score_norm_factor × potential_weight
```

直觀解釋：
- `shanten_number()` 是手牌距離聽牌還差幾步（向聽數）。0 表示已聽牌，數字越大越遠。
- `ukeire`（有効牌枚数）是「現在摸到哪些牌可以讓向聽數前進」，並扣除場上已經被打出或被吃碰走的牌。
- 已聽牌給固定高分，還沒聽牌則用 `ukeire / (shanten+1)` 作為「潛力分數」。

這個設計的好處是：不只告訴模型「你現在離聽牌多遠」，還告訴它「你的手牌有多少成長空間」。Ukeire 大的手牌即使向聽數較高，也可能比 ukeire 小的低向聽手牌更有價值。

然後 `potential_weight` 就是我們講的**「油門」**：
- 攻擊模式：0.4，全力推進
- 防守模式：0.01，40 倍衰減，幾乎斷電

`

## 投影片 7：④ r_penalty —— 放銃防守懲罰（Mode-Aware）

這是我們**最重的懲罰信號**：

```
if 放銃（check_houjuu = True）:
    r_penalty = -|opponent_score_delta| × penalty_weight / score_norm_factor
else:
    r_penalty = 0.0
```

`check_houjuu()` 的實作值得特別說明。放銃判斷不是簡單看誰胡牌——你要回溯確認「是不是你打出的那張牌導致了 RON」。我們的邏輯：

1. 遍歷所有 events，找到 RON 事件
2. 從 RON 事件往前回溯，找到最近一次的 DISCARD / TSUMOGIRI / ADDED_KAN / OPEN_KAN
3. 如果那個動作是你做的 → 確認放銃

特別注意我們也處理了**槍槓（ADDED_KAN / OPEN_KAN 被 RON）**的邊界情況，防止漏抓。

懲罰倍率：
- 攻擊模式：1.8（標準避險警覺）
- 防守模式：**6.0**（3.3 倍暴增）—— 這是我們的「安全氣囊」，防守模式下一放銃就直接重罰，強迫模型選安全牌

---

## 投影片 8：⑤ r_backward —— 終局回溯得分（Mode-Invariant）

這是唯一跟模式無關的 reward 元件，因為它只在你胡牌時觸發：

```
score_basis = score_delta / 1000       （例如 8000 點 → 基底 8）
unit_score = score_basis² / (total_tiles × score_norm_factor)
overlap = min(final_hand_34, current_hand_34)   （逐 tile 取最小）
r_backward = Σ(overlap × unit_score)
```

**核心思想**：你最終胡牌得分，不應該只歸功於最後一步「宣告 TSUMO/RON」——整局中每一步保留關鍵牌的決策都有貢獻。

所以我們把最終得點平方後（拉開大牌與小牌的差距），依照**每一步手牌與最終胡牌面的重疊程度（overlap）**，反向分配到軌跡（trajectory）的每一步。

這個平方設計經過數值安全考量：如果 raw score 是 8000，平方後 64,000,000 會讓 Critic 爆炸。所以我們先把點數除以 1000（8² = 64），再除以 tile 數量（14）和 norm factor（10000），最終 unit_score 落在非常安全的範圍內。

---

## 投影片 9：攻守參數對比總表

| 參數 | ⚔️ Attack | 🛡️ Defense | 倍率變化 | 設計意圖 |
|------|----------|-----------|---------|---------|
| `penalty_weight` | 1.8 | **6.0** | 3.3× ↑ | Defense 放銃 = 死刑 |
| `potential_weight` | 0.4 | **0.01** | 40× ↓ | Defense 關閉進攻油門 |
| `dora_weight` | 0.01 | **0.002** | 5× ↓ | Defense 允許犧牲寶牌 |
| `progression_weight` | 0.05 | **0.005** | 10× ↓ | Defense 不因進展而冒險 |

我們可以用一句話總結：

> **Attack = 油門全開；Defense = 油門斷電 + 安全氣囊**

---

## 投影片 10：在訓練流程中的實際運作

最後快速走一遍 reward 在訓練 pipeline 中如何被使用：

**Step 1：初始化（`SelfPlayRunner.__init__`）**
```python
self.reward_calculator = create_default_calculator()
self.reward_calculator.set_mode(train_mode)  # "attack" or "defense"
```
全程固定，不切換。

**Step 2：每步計算即時 reward（`run_match` 主迴圈）**
```python
step_reward  = r_potential   # 向聽潛力
step_reward += r_dora        # 寶牌留存
step_reward += r_progression # 向聽進展（delta × progression_weight）
```
每一步約 0.2–0.5 的 reward，Critic 持續有訊號。

**Step 3：終局後處理**
```python
if 放銃:    最後一步 reward += r_penalty    （負值，-0.5 ~ -5.0）
if 胡牌:    每一步 reward += r_backward     （正向分配，逐步累積）
if 沒胡:    每一步 reward += -0.001         （微小懲罰，鼓勵胡牌）
```

**Step 4：進入 PPO（`train_ppo_step.py`）**
```
step_rewards → GAE (γ=0.99, λ=0.95) → Advantages → Normalize
Advantages + Values → Returns → Value Loss (MSE)
Advantages → PPO Clipped Surrogate Loss
```

GAE 的 λ-return 會把即時 reward 和終局 reward 的訊號在時間維度上平滑擴散，讓每一條軌跡的每一步都有合理的 advantage estimate。

---

## 投影片 11：總結與未來方向

**這套 reward 系統解決了三個核心問題：**

1. **稀疏獎勵 → 稠密梯度**：五項 reward 元件讓每一步都有 learning signal，Critic 穩定學習
2. **攻守抉擇**：雙模式分離訓練，Attack 專攻、Defense 專守，各自最優化策略
3. **Credit Assignment**：r_backward 回溯分配，讓早期保留關鍵牌的步驟也能得到合理獎勵

**架構上的優雅之處：**
- 所有權重集中在 `MODE_PARAMS` 字典中，消融實驗只需改動一處
- 模式控制透過 `set_mode()` + property 動態讀取，代碼乾淨無重複
- 與 PPO 訓練循環解耦，reward calculator 是獨立模組

**未來可以探索的方向：**
- 動態模式切換：在單一模型中透過 Condition Injection 實現攻守人格動態切換
- 更多模式：如「速攻」（追求最快聽牌）、「大牌」（追求役滿）
- 對手建模：根據對手行為自動調整 reward 權重

---

> 以上，謝謝各位。歡迎提問。