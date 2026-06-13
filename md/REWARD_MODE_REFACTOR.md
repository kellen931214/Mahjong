
## 雙模型分離訓練

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



