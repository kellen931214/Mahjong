# PPO 訓練版本對比：v2 → v3

> 訓練腳本：`train_ppo.py`  
> Reward 模組：`rewards.py`  
> 對手產生：`runner.py`

---

## 版本總覽

| 項 | v2 (`ppo_lora_v2`) | v3 (`ppo_lora_v3`) — 方案四 |
|---|-------------------|---------------------------|
| **總獎勵公式** | `r_potential + r_progression + r_backward + r_penalty` | `r_potential + r_progression + r_dora + r_backward + r_penalty` |
| **r_backward 計算** | 線性：`final_score / total_tiles` | **非線性**：`final_score × √final_score / (total_tiles × norm)` |
| **r_penalty 計算** | `−對手得分 / norm` | **×2 倍權重**：`−對手得分 × 2.0 / norm` |
| **r_dora** | ❌ 無 | 🆕 `dora_count × 0.5 / norm`（每步即時獎勵） |
| **Checkpoint 目錄** | `checkpoints/ppo_lora_v2/` | `checkpoints/ppo_lora_v3/` |

---

## 🆕 方案四詳細說明

### 1. r_backward：非線性分數獎勵（鼓勵做大牌）

v2 的線性公式無法區分小牌和大牌：
- 1000 點（1 翻）= reward ≈ 0.07
- 8000 點（4 翻）= reward ≈ 0.57
- 兩者只差 8 倍，但實際牌型難度差很多

v3 使用 **分數²** 非線性放大：
- 45 分（1000 點）→ reward ≈ **2.0**（每張牌）
- 90 分（2000 點）→ reward ≈ **8.1**（每張牌，4x 差距）
- 360 分（7700 點）→ reward ≈ **129.6**（每張牌，65x 差距）

→ 大幅提高高分牌的期望值，降低「快速胡小牌」的相對吸引力。

### 2. r_penalty：加倍放銃懲罰（鼓勵防守）

v2 放銃懲罰較輕，模型可能為追求速度而打出危險牌。

v3 將懲罰倍率從 1x 提高到 **2x**，強化防守意識：
- 放銃給對手 90 分 → v2 扣 0.09 / v3 扣 0.18

### 3. r_dora：寶牌即時獎勵（引導高價值路線）

每多一張寶牌，每步獲得 0.0005 的即時獎勵：
- 0 張寶牌 → +0.0000
- 2 張寶牌 → +0.0010
- 5 張寶牌 → +0.0025

這會讓模型在聽牌過程中傾向「保留寶牌」的特殊方向，自然走向高翻數路線。

---

## 預期效果

| 指標 | v2 現狀 | v3 預期變化 |
|------|--------|------------|
| **和了率** | 54~68% | ⬇ 輕微下降（不再盲目追求快速和牌） |
| **贏牌率** | 28~30% | ⬆ 上升（和牌質量提高 → 拿第一名比例增加） |
| **均排名** | 2.36 | ⬆ 趨向 2.0~2.2 |
| **平均打點** | 偏低（多為 1~2 翻） | ⬆ 明顯提高 |

---

## 運行方式

```bash
# v3 訓練（自動使用 ppo_lora_v3 checkpoint 目錄）
python train_ppo.py

# 若要調整方案四的超參數：
python train_ppo.py --checkpoint-dir ./checkpoints/ppo_lora_v3_custom
```

v2 的 checkpoint 和 log 完整保留在 `checkpoints/ppo_lora_v2/`，不受影響。