# ⚔️🛡️ Mahjong Decision Mamba — 攻守雙模式 Reward 重構報告

> 版本：v3.0（雙模型分離訓練）
> 日期：2026-06-03

---

## 方案一（參考）：RIICHI 動態切換（單模型 + 特徵注入）

> 一個模型，根據 `obs.events()` 中是否出現 `EventType.RIICHI` 自動切換 reward 權重

### 架構圖

```
obs.events() 掃描 RIICHI
       │
       ▼
  set_mode("attack" / "defense")
       │
       ├──→ 動態權重讀取（_pw / _ptw / _dw / _pgw）
       │
       └──→ 2D One-hot 拼接 → 1382 維 augmented state → Mamba
```

### 概念程式碼（runner.py 內的 step）

```python
# 步驟 1：檢查有無人立直
for event in obs.events():
    if event.type() == EventType.RIICHI:
        reward_calculator.set_mode("defense")
        break
else:
    reward_calculator.set_mode("attack")

# 步驟 2：2D One-hot 特徵注入
raw_state_1380 = obs.to_features("decision-mamba-v0")
mode_tensor = [1.0, 0.0] if mode == "attack" else [0.0, 1.0]
augmented_state = np.concatenate([raw_state_1380, mode_tensor])  # 1382 dims
```

### ⚠️ 缺點

- 需要 1382 維模型（BC checkpoint 1380→1382 權重擴展）
- Reward 在 episode 中跳變 → Critic 學習困難
- SSM 需同時學會兩種策略 → 訓練不穩定

---

## 方案二（✅ 採用）：雙模型分離訓練

> 兩個模型各自訓練：Attack Model 永遠只用進攻 reward，Defense Model 永遠只用防守 reward

### 訓練指令

```bash
# 訓練攻擊型模型（預設）
python train_ppo.py --mode attack

# 訓練防守型模型
python train_ppo.py --mode defense
```

### ⚔️ 攻擊模式（Attack Mode）

| # | Reward 元件 | 權重 | 公式 |
|---|-----------|------|------|
| 1 | **r_potential**（向聽潛力） | `0.4` | base × ukeire/(shanten+1) / 10000 × 0.4 |
| 2 | **r_dora**（寶牌留存） | `0.01` | dora_count × 0.01 |
| 3 | **r_progression**（向聽進展） | `0.05` | (prev_shanten − curr_shanten) × 0.05 |
| 4 | **r_penalty**（放銃懲罰） | `1.8` | −|opponent_score| × 1.8 / 10000 |
| 5 | **r_backward**（終局回溯得分） | 不變 | (score/1000)² / (tiles × 10000) × overlap |

### 🛡️ 防守模式（Defense Mode）

| # | Reward 元件 | 權重 | 公式 |
|---|-----------|------|------|
| 1 | **r_potential**（向聽潛力） | `0.01` | 油門斷電，40× 衰減 |
| 2 | **r_dora**（寶牌留存） | `0.002` | 可安全切出寶牌，5× 衰減 |
| 3 | **r_progression**（向聽進展） | `0.005` | 不鼓勵冒進，10× 衰減 |
| 4 | **r_penalty**（放銃懲罰） | `6.0` | 極重放銃懲罰，3.3× 暴增 |
| 5 | **r_backward**（終局回溯得分） | 不變 | 防守成功摺牌則自然為 0 |

### 權重對比一覽

| 參數 | ⚔️ 攻擊 | 🛡️ 防守 | 倍率變化 |
|------|--------|--------|---------|
| `penalty_weight` | 1.8 | **6.0** | 3.3× 暴增 |
| `potential_weight` | 0.4 | **0.01** | 40× 衰減 |
| `dora_weight` | 0.01 | **0.002** | 5× 衰減 |
| `progression_weight` | 0.05 | **0.005** | 10× 衰減 |

### 架構圖

```
BC checkpoint (1380 dims)
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

---

## 方案一 vs 方案二 對比

| 維度 | 方案一（RIICHI 動態切換） | 方案二（雙模型分離訓練） |
|------|------------------------|------------------------|
| 模型數量 | 1 個（學兩種策略） | 2 個（各司其職） |
| 狀態維度 | 1382（需特徵注入） | 1380（與 BC 完全相容） |
| Reward 平穩性 | ❌ 中途跳變 | ✅ 全程固定 |
| Critic 穩定性 | ⚠️ 方差大 | ✅ 最穩定 |
| 訓練方式 | 單次 | 兩次（attack + defense） |
| BC 相容性 | 需權重擴展 | 直讀直寫 |
| 消融實驗 | 複雜（需分離模式貢獻） | 乾淨二元對比 |
| 論文敘事力 | Phase Transition | 攻守解耦 |

---

## 檔案變更清單

| 檔案 | 改動 |
|------|------|
| `utli/rewards.py` | `MODE_PARAMS` 雙參數字典、`set_mode()`、動態 property |
| `utli/runner.py` | `train_mode` 建構時固定模式、純 1380 維 |
| `model.py` | `state_dim=1380` |
| `train_ppo.py` | `--mode attack\|defense` CLI |
| `evaluate.py` | `state_dim=1380`、`--eval-mode attack\|defense` |