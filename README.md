# 🀄 Mahjong AI — 基於 Decision Mamba 的日本麻將強化學習專案

> 使用 **Mamba State Space Model** + **LoRA 低秩適配** + **PPO 自我博弈**，從人類對局資料進行行為模仿，再透過雙模式攻守獎勵進行線上微調的日麻 AI。

---

## 📋 目錄

1. [專案概述](#1-專案概述)
2. [環境與依賴](#2-環境與依賴)
3. [資料處理流程](#3-資料處理流程)
4. [模型架構詳解](#4-模型架構詳解)
5. [第一階段：BC 行為模仿預訓練](#5-第一階段bc-行為模仿預訓練-train_bcpy)
6. [第一階段延伸：多頭分權 + Focal Loss 微調](#6-第一階段延伸多頭分權--focal-loss-微調-train_bc_multheadpy)
7. [LoRA 低秩適配機制](#7-lora-低秩適配機制)
8. [第二階段：PPO + LoRA 自我博弈微調](#8-第二階段ppo--lora-自我博弈微調-train_ppopy)
9. [Reward 獎勵設計詳解](#9-reward-獎勵設計詳解)
10. [PPO 演算法細節](#10-ppo-演算法細節)
11. [評估系統](#11-評估系統)
12. [訓練與評估指令](#12-訓練與評估指令)
13. [專案目錄結構](#13-專案目錄結構)
14. [參考文獻](#14-參考文獻)

---

## 1. 專案概述

### 1.1 目標

打造一個能在日本麻將（Riichi Mahjong）中達到競技水準的 AI 代理，透過兩階段訓練策略：
- **第一階段**：從人類專家對局資料中學習基礎牌效（Behavioral Cloning）
- **第二階段**：透過自我博弈（Self-Play）與 PPO 進行強化學習微調，提升實戰決策能力

### 1.2 技術棧

| 層級 | 技術 |
|------|------|
| 模型骨幹 | **Mamba** (State Space Model) — `mamba-ssm` |
| 架構設計 | **Decision Mamba**（RTG-conditioned sequence model） |
| 微調技術 | **LoRA** (Low-Rank Adaptation) |
| 損失函數 | **Focal Loss** + **Multi-Head CrossEntropy** |
| 遊戲引擎 | **mjx** (C++ 高效麻將模擬器 + Python binding) |
| 強化學習 | **PPO** (Proximal Policy Optimization) + **GAE** |
| 對手 AI | **Mortal**（外部最強開源日麻 AI） |
| 評估體系 | **Suphx** 論文指標（per-hand 和了率 / 放銃率 / 順位分佈） |

### 1.3 整體架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                    第一階段：BC 預訓練                         │
│                                                             │
│  人類對局紀錄 (mjai JSON)                                     │
│       │                                                     │
│       ▼                                                     │
│  convert/ 特徵轉換 → .npy chunks (1380-dim + 181-dim)        │
│       │                                                     │
│       ▼                                                     │
│  BehavioralCloningDataset → DataLoader                      │
│       │                                                     │
│       ▼                                                     │
│  DecisionMamba (單頭) ──→ train_bc_step()                   │
│  Multi-task Loss: λ₁·CE + λ₂·MSE(RTG) + λ₃·MSE(State)      │
│       │                                                     │
│       ▼                                                     │
│  BC checkpoint                                              │
│       │                                                     │
│       ├──→ 多頭分權微調 (train_bc_multhead.py)               │
│       │    DecisionMambaMultiHead + Focal Loss               │
│       │    凍結 Backbone，僅訓練 5 個專職 Head                │
│       │                                                     │
└───────┼─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              第二階段：PPO + LoRA 自我博弈                     │
│                                                             │
│  載入 BC/MultiHead checkpoint                                │
│       │                                                     │
│       ▼                                                     │
│  prepare_for_ppo() → 凍結 Backbone                           │
│  僅訓練: lora_A/B + Actor Head + Critic Head + State Head    │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────────────┐                │
│  │       SelfPlayRunner (mjx 環境)          │                │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐ │                │
│  │  │Agent │  │Opp 1 │  │Opp 2 │  │Opp 3 │ │                │
│  │  │(PPO) │  │(Pool)│  │(Pool)│  │(Pool)│ │                │
│  │  └──────┘  └──────┘  └──────┘  └──────┘ │                │
│  └─────────────────────────────────────────┘                │
│       │                                                     │
│       ▼                                                     │
│  train_ppo_step() → GAE → PPO Clip → Backward               │
│  五項 Reward 元件 (攻/守雙模式)                                │
│       │                                                     │
│       ▼                                                     │
│  PPO checkpoint → evaluate.py                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 環境與依賴

### 2.1 Python 環境

- Python ≥ 3.9
- PyTorch ≥ 2.0（CUDA 支援）
- CUDA ≥ 11.8

### 2.2 核心依賴

| 套件 | 用途 |
|------|------|
| `torch` | 深度學習框架 |
| `mamba-ssm` | Mamba State Space Model 實作 |
| `numpy` | 數值計算 |
| `mjx` | 日麻遊戲引擎（C++ + pybind，本專案子模組） |
| `libriichi` | Mortal 的 Rust 麻將引擎（透過 maturin 安裝） |

### 2.3 安裝步驟

```bash
# 1. 克隆專案（含子模組 mjx 和 Mortal）
git clone --recurse-submodules https://github.com/kellen931214/Mahjong.git
cd Mahjong

# 2. 安裝 mjx
cd mjx && pip install -e . && cd ..

# 3. 安裝 Mortal 依賴（libriichi 需透過 maturin 安裝）
pip install maturin
cd Mortal/libriichi && maturin develop --release && cd ../..

# 4. 安裝其他依賴
pip install torch mamba-ssm numpy
```

---

## 3. 資料處理流程

### 3.1 原始資料來源

- **格式**：mjai JSON 格式的日本麻將對局紀錄（每局為一系列事件流）
- **內容**：包含 start_game、start_kyoku、tsumo、dahai、chi、pon、kan、reach、hora、ryukyoku 等事件

### 3.2 特徵轉換（`convert/` 目錄）

原始 mjai 事件流經過 `convert/convert_mjson_to_features.py` 轉換為模型可用的數值格式：

**狀態特徵（1380 維）**：
- **空間特徵**（1360 維）：40 通道 × 34 牌種
  - ch 0~3：自家手牌（依持有張數分通道）
  - ch 4~7：自家副露（吃/碰/槓）
  - ch 8~19：三家副露
  - ch 20~35：四家牌河（每家 4 段，每段 6 張）
  - ch 36~39：寶牌指示牌
- **標量特徵**（20 維）：四人分數、剩餘牆牌比例、場風、局數、本場數、立直棒數、立直狀態、莊家指示

**動作標籤（181 維編碼，0~180）**：

| 區間 | 動作類型 | 維度 |
|------|---------|------|
| 0~36 | DISCARD（手切） | 37 |
| 37~73 | TSUMOGIRI（摸切） | 37 |
| 74~103 | CHI（吃） | 30 |
| 104~140 | PON（碰） | 37 |
| 141~174 | KAN（槓，含暗槓/大明槓/加槓） | 34 |
| 175 | TSUMO（自摸） | 1 |
| 176 | RON（榮和） | 1 |
| 177 | RIICHI（立直） | 1 |
| 178 | NINE_TERMINALS（九種九牌） | 1 |
| 179 | NO（無動作/跳過） | 1 |
| 180 | DUMMY（虛擬填充） | 1 |

### 3.3 軌跡分割與儲存

- 轉換後的資料以 **對局為單位**（一條完整半莊）進行軌跡分割
- 儲存為 **未壓縮 `.npy` 格式**，分散在多個 `chunk_*/` 子目錄中
- 每個 chunk 包含三個矩陣：
  - `features.npy`：形狀 `(N, 1380)` 的 float32 狀態序列
  - `actions.npy`：形狀 `(N,)` 的 int64 動作序列
  - `rtgs.npy`：形狀 `(N,)` 的 float32 return-to-go 序列
  - `trajectory_boundaries.npy`：各軌跡的邊界索引

### 3.4 資料載入（`dataset.py`）

`BehavioralCloningDataset` 採用 **mmap 延遲載入**策略：

1. **主進程**：僅掃描 `trajectory_boundaries.npy`，記錄軌跡的 `(chunk_idx, start, end)` 索引，不載入實際資料
2. **子進程（DataLoader workers）**：首次 `__getitem__` 時以 `np.load(..., mmap_mode='r')` 建立進程專屬的記憶體映射
3. **LRU 快取**：每個 worker 快取最近 8 個 chunk 的 mmap 物件（OrderedDict），命中時直接切片
4. **序列截斷**：單條軌跡超過 512 步時，保留最後 512 步（`start = end - 512`）

### 3.5 批次處理與自迴歸右移

`bc_collate_fn` 負責將不等長軌跡填充為批次：

```
原始 input_action:  [a₁, a₂, a₃, ..., a_T]    (target 標籤保持原樣)
安全 input_action:  [180, a₁, a₂, ..., a_{T-1}] (右移一格，開頭補 DUMMY)
```

這個 **自迴歸 Shift 機制** 防止模型在預測 `a_t` 時偷看到 `a_t` 本身（Look-Ahead Leakage）。模型在 step t 的輸入是 `[s_t, a_{t-1}, rtg_t]`，預測 `a_t`。

---

## 4. 模型架構詳解

### 4.1 DecisionMamba（`model.py`）

```
                    ┌──────────┐  ┌──────────┐  ┌──────────┐
    rtg (B,T,1) ──→│ Embed    │  │ Embed    │  │ Embed    │
                    │ RTG      │  │ State    │  │ Action   │
                    │ (1→512)  │  │(1380→512)│  │(181→512) │
                    └────┬─────┘  └────┬─────┘  └────┬─────┘
                         │             │             │
                         └──────┬──────┴──────┬──────┘
                                │  Concat     │
                                │ (B,T,1536)  │
                                └──────┬──────┘
                                       │
                                ┌──────▼──────┐
                                │  Input Proj │
                                │ (1536→512)  │
                                └──────┬──────┘
                                       │ + Timestep Embedding
                                       ▼
                                ┌──────────────┐
                                │MultiGrained  │
                                │    Block     │ ← 含 LoRA 注入
                                └──────┬───────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
              ┌─────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐
              │ Action     │    │ RTG Head    │   │ State Head  │
              │ Head (181) │    │ (512→1)     │   │ (512→1380)  │
              └───────────┘    └─────────────┘   └─────────────┘
```

**超參數**：
- `d_model` = 512（隱藏層維度）
- `action_dim` = 181（mjx 完整動作空間）
- `state_dim` = 1380（特徵維度）
- `max_ep_len` = 2048（最大 timestep embedding 範圍）

### 4.2 MultiGrainedBlock（雙粒度 Mamba 區塊）

```
                    h_{i-1} (B, T, 512)
                         │
                    ┌────▼────┐
                    │LayerNorm│
                    └────┬────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │Conv1d Coarse│ │Conv1d Fine │ │  Proj z_cg │  ← LoRALinear
   │(kernel=3)   │ │(kernel=3)  │ │  Proj z_fg │  ← LoRALinear
   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
         │              │              │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │  SiLU   │    │  SiLU   │    │  SiLU   │
    └────┬────┘    └────┬────┘    └────┬────┘
         │              │              │
    ┌────▼────┐    ┌────▼────┐         │
    │Mamba CG │    │Mamba FG │         │
    │(d_state │    │(d_state │         │
    │ =16)    │    │ =16)    │         │
    └────┬────┘    └────┬────┘         │
         │              │              │
         │    h_cg      │    h_fg      │ gate
         └──────┬───────┘      │       │
                │   × (gating) │       │
                ▼              ▼       │
           h_cg_gated    h_fg_gated    │
                │              │       │
                └──────┬───────┘       │
                       ▼               │
                 ┌──────────┐          │
                 │   Add    │←─────────┘
                 └────┬─────┘
                      │
                 ┌────▼─────┐
                 │LayerNorm │ (fusion_norm)
                 └────┬─────┘
                      │
                 ┌────▼─────┐
                 │ Out Proj │ ← LoRALinear
                 └────┬─────┘
                      │ + h_{i-1} (residual)
                      ▼
                    h_i
```

每個 MultiGrainedBlock 包含：
- **雙粒度 Conv1d**：粗粒度（CG）和細粒度（FG），使用相同的 kernel_size=3, dilation=2
- **雙 Mamba SSM**：`d_state=16`, `d_conv=4`, `expand=2`
- **Gating 機制**：CG/FG 輸出與 SiLU(proj_z) 做 element-wise 乘積
- **Fusion**：CG 和 FG 分支相加後經 LayerNorm 融合
- **LoRA 注入**：proj_z_cg、proj_z_fg、out_proj 均為 LoRALinear

### 4.3 DecisionMambaMultiHead（多頭分權版本）

將原始的單一 `nn.Linear(512, 181)` Action Head 替換為 **評審團拼接機制（Jury Concatenation）**：

```
                    h (B, T, 512)
                         │
    ┌────────┬───────────┼───────────┬──────────┐
    ▼        ▼           ▼           ▼          ▼
┌──────┐ ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│Discard│ │ Chow │   │ Pong │   │ Kong │   │Special│
│512→74 │ │512→30│   │512→37│   │512→34│   │512→6 │
└──┬───┘ └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘
   │        │          │         │          │
   └────────┴──────────┴─────────┴──────────┘
                      │
                torch.cat(dim=-1)
                      │
                  (B, T, 181)
```

五個專職 Linear Head 與 mjx 動作編碼的對應：
- **Discard Head** (74 dims)：0~73 — DISCARD + TSUMOGIRI
- **Chow Head** (30 dims)：74~103 — CHI（含赤牌變體）
- **Pong Head** (37 dims)：104~140 — PON（含赤牌變體）
- **Kong Head** (34 dims)：141~174 — CLOSED/OPEN/ADDED KAN
- **Special Head** (6 dims)：175~180 — TSUMO/RON/RIICHI/NINE_TERMINALS/NO/DUMMY

---

## 5. 第一階段：BC 行為模仿預訓練（`train_bc.py`）

### 5.1 訓練目標

讓模型學習模仿人類專家的打牌決策。這是一個**多任務學習**問題：

```
Total Loss = λ₁ × Action Loss  +  λ₂ × RTG Loss  +  λ₃ × State Loss
           = 0.6 × CE           +  0.3 × MSE      +  0.1 × MSE
```

### 5.2 訓練資料

- **輸入**：`/data/converted_features_npy/` 下的 `chunk_*/` 目錄
- **驗證集比例**：20%（與訓練集使用相同的 `random_split`，seed=42）
- **批次大小**：256（預設）
- **DataLoader workers**：4（NVMe 環境）
- **序列最大長度**：512 steps（超過則截斷尾部）

### 5.3 訓練步驟（`utli/train_bc_step.py`）

```python
def train_bc_step(model, batch, optimizer, lambdas=(0.6, 0.3, 0.1)):
    # 1. 前向傳播：model(rtg, state, input_action, timesteps)
    #    - input_action 已在 collate_fn 中完成自迴歸右移
    pred_action, pred_rtg, pred_state, _ = model(...)

    # 2. 動作預測損失（CrossEntropy，ignore_index=-100 過濾 padding）
    ce_loss = F.cross_entropy(pred_action.reshape(-1, 181),
                               target_action.reshape(-1),
                               ignore_index=-100)

    # 3. RTG 預測損失（MSE，僅有效時間步）
    rtg_loss = MSE(pred_rtg, rtg) * valid_mask

    # 4. 狀態預測損失（MSE，僅非零空間特徵 + 所有標量特徵）
    state_loss = MSE(pred_state, state) * state_mask

    # 5. 加權總損失
    total_loss = 0.6 * ce_loss + 0.3 * rtg_loss + 0.3 * state_loss

    # 6. 反向傳播（梯度裁剪 max_norm=1.0）
    total_loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### 5.4 訓練超參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `num_epochs` | 100 | 訓練輪數 |
| `learning_rate` | 1e-3 | AdamW 學習率 |
| `weight_decay` | 1e-4 | 權重衰減 |
| `lr_scheduler` | CosineAnnealingLR | T_max = num_epochs |
| `batch_size` | 256 | 批次大小 |
| `val_split` | 0.2 | 驗證集比例 |
| `seed` | 42 | 隨機種子 |

### 5.5 監控指標

- **Action Accuracy**：排除 padding (-100) 和 NO action (179) 後的正確率
- **RTG Loss**：return-to-go 預測的 MSE
- **State Loss**：狀態重建的 MSE
- **最佳模型選擇**：最低驗證損失

### 5.6 啟動指令

```bash
python train_bc.py \
    --npz-file /data/converted_features_npy \
    --batch-size 256 \
    --num-epochs 100 \
    --learning-rate 1e-3 \
    --device cuda
```

---

## 6. 第一階段延伸：多頭分權 + Focal Loss 微調（`train_bc_multhead.py`）

### 6.1 為什麼需要多頭微調？

原始 BC 模型使用單一 `nn.Linear(512, 181)` 輸出層。這導致兩個問題：
1. **類別極度不平衡**：Dahai（切牌）佔 90%+ 的動作，Kong（槓）可能 < 0.5%，模型容易忽略稀有動作
2. **共享梯度衝突**：不同動作類型（切牌 vs 吃碰槓）的梯度方向可能互相矛盾

### 6.2 解決方案：Jury Concatenation + Focal Loss

- **5 個專職 Head**：每個 Head 獨立學習其對應動作類別的決策邊界
- **Focal Loss**：自動壓制高頻易分類動作（Dahai）的梯度，放大低頻難分類動作（Kong）的梯度

### 6.3 Focal Loss 數學原理（`utli/focal_loss.py`）

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

其中：
  p_t         = softmax(logits)[target_class]  （模型對真實類別的預測機率）
  (1 - p_t)^γ = 非線性調製係數（Modulating Factor）
  α_t         = 類別平衡權重（Class-Balancing Weight）
  γ           = 聚焦指數（Focusing Parameter, 預設 2.0）
```

**γ = 2.0 的效果**：
| 場景 | p_t | focal_weight = (1-p_t)² | 梯度衰減 |
|------|-----|--------------------------|---------|
| Dahai 易分類 | 0.9 | 0.01 | **100× 衰減** → 不再浪費容量 |
| Kong 難分類 | 0.1 | 0.81 | **僅 19% 衰減** → 學習訊號完整保留 |

**α_t 類別加權**：
| 動作區間 | α 值 | 說明 |
|---------|------|------|
| Dahai (0~73) | 1.0 | 基線 |
| Chow (74~103) | 1.0 | 基線 |
| Pong (104~140) | 1.0 | 基線 |
| **Kong (141~174)** | **3.0** | **強行放大稀有槓牌梯度** |
| Special (175~180) | 1.0 | 基線 |

### 6.4 微調初始化流程（`setup_multihead_finetune`）

```
Step 1: 實例化 DecisionMambaMultiHead
Step 2: 載入 BC 預訓練權重（strict=False）
        → 手動清除舊的 head_action.weight/bias
        → 骨幹參數自動對齊
Step 3: 骨幹凍結（Backbone Freezing）
        → 所有 embedding / input_proj / block / head_rtg / head_state 凍結
        → 僅 head_action.head_discard/head_chow/head_pong/head_kong/head_special 可訓練
Step 4: AdamW 過濾器（僅收集 requires_grad=True 的參數）
Step 5: 可選 Focal Loss 初始化
```

### 6.5 啟動指令

```bash
# 標準 CrossEntropy 微調（預設）
python train_bc_multhead.py \
    --pretrained checkpoints/bc_model/best_bc_model.pt \
    --npz-file /data/converted_features_npy \
    --num-epochs 30 \
    --learning-rate 5e-4

# Focal Loss 微調（γ=2.0, Kong=3.0×）
python train_bc_multhead.py \
    --pretrained checkpoints/bc_model/best_bc_model.pt \
    --npz-file /data/converted_features_npy \
    --loss-mode focal \
    --focal-gamma 2.0 \
    --kong-weight 3.0
```

---

## 7. LoRA 低秩適配機制

### 7.1 LoRA 原理

LoRA（Low-Rank Adaptation）在凍結的預訓練權重旁邊注入可訓練的低秩矩陣：

```
原始：  h = W · x          （W 凍結，shape: (d, d)）
LoRA：  h = W · x + (x @ A @ B) × (α/r)

其中：
  A: (d, r)  — 低秩矩陣（Kaiming 初始化）
  B: (r, d)  — 低秩矩陣（零初始化）
  r: rank（預設 8）
  α: scaling factor（預設 16）
```

### 7.2 本專案中的 LoRA 注入點

在 `MultiGrainedBlock` 中，以下三個 Linear 層使用 `LoRALinear`：
- `proj_z_cg`：粗粒度 gating 投影
- `proj_z_fg`：細粒度 gating 投影
- `out_proj`：區塊輸出投影

### 7.3 LoRA 喚醒微調（`utli/lora_focal_finetune.py`）

`setup_lora_focal_finetune()` 實現四階段初始化：

```
Stage 1: 全體凍結（Freeze All）
    → 所有參數 requires_grad = False

Stage 2: 選擇性解凍（Selective Unfreeze）
    → 僅解凍 lora_A, lora_B（低秩矩陣）
    → 僅解凍 head_action.*（多頭輸出層）
    → 不解凍 head_rtg.* 和 head_state.*

Stage 3: Focal Loss 建構
    → alpha_weights (181,) with Kong=3.0×
    → MahjongFocalLoss(γ=2.0)

Stage 4: AdamW 過濾器
    → filter(lambda p: p.requires_grad, model.parameters())
    → lr=3e-4（比預訓練 5e-4 保守）
```

### 7.4 PPO 階段的 LoRA（`prepare_for_ppo()`）

與 LoRA 喚醒微調不同，PPO 階段解凍範圍更廣：

```python
def prepare_for_ppo(self):
    # Stage 1: 全部凍結
    for param in self.parameters():
        param.requires_grad = False

    # Stage 2: 選擇性解凍
    for name, param in self.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True       # LoRA 矩陣
        elif name.startswith("head_action.") \
          or name.startswith("head_rtg.") \
          or name.startswith("head_state."):
            param.requires_grad = True       # 三個輸出 Head（PPO 需要 Critic）
```

PPO 階段**同時解凍 head_rtg 和 head_state**，因為 Critic 需要重新學習 value function。

---

## 8. 第二階段：PPO + LoRA 自我博弈微調（`train_ppo.py`）

### 8.1 訓練流程

```
for iteration in 1..num_iterations:
    ┌─────────────────────────────────────────┐
    │ 1. 自我博弈收集軌跡 (runner.run_match)   │
    │    - 4 人 mjx 環境                       │
    │    - Agent 使用當前模型 (採樣模式)         │
    │    - 3 個對手隨機來自對手池                │
    │    - 收集 (obs, action, reward, log_prob) │
    │    - 每 iter 收集 8 條軌跡後合併更新       │
    └──────────────────┬──────────────────────┘
                       ▼
    ┌─────────────────────────────────────────┐
    │ 2. PPO 更新 (train_ppo_epoch)            │
    │    - GAE 計算 advantage                   │
    │    - 標準化 advantage                     │
    │    - 計算 PPO clipped objective           │
    │    - Value loss + Entropy bonus           │
    │    - 反向傳播 (僅 LoRA + Heads)           │
    └──────────────────┬──────────────────────┘
                       ▼
    ┌─────────────────────────────────────────┐
    │ 3. 更新對手池 (每 10 iter)               │
    │    - 複製當前模型加入對手池               │
    │    - 超出容量時移除最舊的非當前模型         │
    └──────────────────┬──────────────────────┘
                       ▼
    ┌─────────────────────────────────────────┐
    │ 4. 儲存 checkpoint (每 50 iter)          │
    │    - 模型權重 + optimizer 狀態            │
    │    - 訓練歷史 (policy/value loss 等)     │
    │    - 遊戲指標 (贏牌率、和了率、均排名)     │
    └─────────────────────────────────────────┘
```

### 8.2 PPO 超參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `num_iterations` | 1000 | 訓練迭代次數 |
| `ppo_epochs` | 1 | 每條軌跡的 PPO 更新次數 |
| `learning_rate` | 5e-5 | LoRA 學習率（小步伐） |
| `temperature` | 0.8 | Logit 採樣溫度（低溫加速收斂） |
| `entropy_coef` | 0.01 | 策略熵係數（減少強制探索） |
| `value_coef` | 0.5 | Value Loss 權重 |
| `clip_epsilon` | 0.2 | PPO 裁剪閾值 |
| `max_grad_norm` | 0.5 | 梯度裁剪閾值 |
| `gamma` (GAE) | 0.99 | 折扣因子 |
| `lam` (GAE) | 0.95 | GAE λ 參數 |
| `reward_mode` | `sparse` 或 `dense` | 獎勵模式：sparse=純終局 +1/0/-1，dense=shaping+終局混合 |
| `num_trajectories` | 32 | 每 iter 收集軌跡數（可透過 `--num-trajectories` 調整） |

### 8.3 自我博弈機制（`utli/runner.py`）

**玩家分配**：
- Agent（PPO 訓練中）：`self.model`，使用溫度採樣
- 對手：從 `opponent_pool` 中隨機選取歷史模型快照
- 支援外部 AI 注入（如 MortalAgent），此時其餘三人全部使用外部 AI

**軌跡收集**：
`run_match()` 全程使用 `train_mode` 指定的 reward 權重（attack 或 defense），不回傳動態切換。每步記錄：
- `obs`：1380 維狀態張量
- `action`：選擇的動作索引 (0~180)
- `log_prob`：採樣動作的對數機率
- `reward`：該步的即時 reward
- `timestep`：時間步索引
- `mask`：合法動作 boolean mask (181,)

**每局牌（Hand）獨立統計**：
在每局結束時（`done("round")`），從 `round_terminal` proto 提取胡牌/放銃/流局狀態，以 per-hand 語義累加。

---

## 9. Reward 獎勵設計詳解

### 9.1 五項 Reward 元件

獎勵計算器 `MahjongRewardCalculator`（`utli/rewards.py`）設計了五項獨立的 reward 元件：

| # | 元件 | 型別 | 觸發時機 | 公式 |
|---|------|------|---------|------|
| 1 | **r_potential** | 即時 | 每步 | 向聽潛力獎勵 |
| 2 | **r_dora** | 即時 | 每步 | 寶牌留存獎勵 |
| 3 | **r_progression** | 即時 | 每步 | 向聽進展獎勵 |
| 4 | **r_penalty** | 延遲 | 終局 | 放銃懲罰 |
| 5 | **r_backward** | 延遲 | 終局（胡牌後回溯） | 終局後向得分分配 |

### 9.2 各元件詳細公式

#### r_potential（向聽潛力獎勵）

```python
if shanten <= 0:  # 已聽牌
    return base_exp_score / score_norm_factor  # = 2000 / 10000 = 0.2
else:
    potential = base_exp_score * ukeire / (shanten + 1)
    return potential / score_norm_factor * potential_weight
```

- `base_exp_score`：聽牌期望得點基準（2000 點）
- `ukeire`：有效進張數（扣除場上已可見的牌）
- `shanten`：向聽數（0 = 聽牌，1~6 = 一到六向聽）
- `potential_weight`：攻守模式權重

#### r_dora（寶牌留存獎勵）

```python
dora_count = count_dora_in_hand(obs)  # 手牌+副露中的真實寶牌數
return dora_count * dora_weight
```

- 寶牌計算包含：赤寶牌 + 指示牌轉換的真實寶牌
- 上限 13 張

#### r_progression（向聽進展獎勵）

```python
delta = prev_shanten - curr_shanten  # >0 改善, <0 退後
return delta * progression_weight
```

- 每當向聽數減少 1，給定正向獎勵
- 向聽數增加時給定負向懲罰

#### r_penalty（放銃懲罰）

```python
if check_houjuu(obs):  # 檢查自己是否放銃
    return -abs(opponent_score) * penalty_weight / score_norm_factor
return 0.0
```

- `check_houjuu()` 回溯檢測：從 RON 事件向前搜尋，找最近一次由自己執行的 DISCARD/TSUMOGIRI/ADDED_KAN/OPEN_KAN
- 涵蓋普通放銃 + 槍槓（ADDED_KAN 被榮和）
- **僅在終局觸發**，一次性地加到最後一步的 reward

#### r_backward（終局後向得分分配）

```python
score_basis = score_delta / 1000.0                         # e.g. 8000 → 8.0
unit_score = (score_basis²) / (total_tiles * score_norm_factor)  # 單位牌值
overlap = min(final_hand_34, current_hand_34)               # 手牌與胡牌面的重疊
raw_reward = sum(overlap * unit_score)
```

- **僅在 agent 胡牌時觸發**（TSUMO 或 RON）
- 將終局得分依照手牌與胡牌面的重疊度，**回溯分配給每一步**
- 非胡牌時（流局或他人胡牌），所有步給定 -0.001 微小懲罰

### 9.3 雙模式攻守分離

訓練時透過 `--mode attack` 或 `--mode defense` 選擇固定模式，全程不切換：

| 參數 | ⚔️ Attack（進攻） | 🛡️ Defense（防守） | 倍率變化 |
|------|-------------------|---------------------|---------|
| `penalty_weight` | 1.8 | **6.0** | 3.3× 暴增 |
| `potential_weight` | 0.4 | **0.01** | 40× 衰減 |
| `dora_weight` | 0.01 | **0.002** | 5× 衰減 |
| `progression_weight` | 0.05 | **0.005** | 10× 衰減 |

**攻擊模式**：全力推進向聽數，積極保留寶牌追求大牌，標準放銃警覺。

**防守模式**：向聽油門被物理斷電，放銃懲罰暴增 3.3 倍，強烈逼迫 Policy 選擇安全牌。

---

## 10. PPO 演算法細節

### 10.1 GAE（Generalized Advantage Estimation）

```python
for t in reversed(range(seq_len)):
    next_val = values_all[t + 1]  # 使用 t+1 步的 value 做 bootstrapping
    delta = reward[t] + γ × next_val - values[t]
    gae = delta + γ × λ × gae
    advantages[t] = gae

returns = advantages + values  # TD(λ) return
```

- **Bootstrapping**：利用第 T+1 步（padding 步）的 value prediction 作為終端價值估計
- **遊戲邊界重置**：多局軌跡交界處強制 `gae = 0`，避免跨局價值洩漏
- **標準化**：`advantages = (advantages - mean) / (std + 1e-8)`
- **Return Z-Score 標準化**：`returns_norm = (returns - mean) / (std + 1e-8)` — 讓 Critic 預測目標固定在 ~[-3, +3] 範圍，消除不同 batch 因 return 分佈差異造成的 Value Loss 震盪

### 10.2 PPO Clipped Objective

```python
ratio = exp(new_log_prob - old_log_prob)           # importance sampling ratio
ratio = clamp(ratio, 0, 10)                        # 數值安全裁剪

surr1 = ratio × advantage
surr2 = clamp(ratio, 1-ε, 1+ε) × advantage        # ε = 0.2
policy_loss = -min(surr1, surr2).mean()
```

### 10.3 Value Loss（Return Z-Score Normalization）

```python
returns_norm = (returns - ret_mean) / (ret_std + 1e-8)  # Z-score 標準化
value_loss = MSE(new_values, returns_norm)                # 直接 MSE
```

- 不再依賴 per-batch `ret_std` 自適應縮放（舊版 `value_loss_raw / ret_std` 因 batch 組成不同造成劇烈跳動）
- Return 被標準化後，Critic 預測目標永遠在 ~[-3, +3]，Value Loss 不再因 batch 組成而震盪

### 10.4 Entropy Bonus

```python
# 僅對合法動作計算 entropy（避免大量 "死 token" 低估真實策略集中度）
legal_probs = probs × legal_mask
legal_log_probs = log_probs × legal_mask  # 非法動作 log_prob = 0
entropy = -(legal_probs × legal_log_probs).sum(dim=-1).mean()

total_loss = policy_loss + value_coef × value_loss - entropy_coef × entropy
```

### 10.5 Action Masking

```python
# 使用大有限負數 (-1e9) 取代 -inf，避免 softmax 產生 NaN
masked_logits = torch.where(legal_mask, logits, NEG_INF)
probs = softmax(masked_logits / temperature)  # temperature=2.0 拉平極端分佈
```

### 10.6 多軌跡合併更新

每 iter 收集 `num_trajectories=8` 條軌跡後合併為一個 batch 更新：

```python
# 8 條軌跡 → concat 為一條長序列
all_trajectory_data → obs_sequence (1, total_steps, 1380)
                    → act_sequence (1, total_steps)
                    → reward_sequence (1, total_steps)
                    → ...

# 遊戲交界處 GAE/RTG 重置
boundary_set = {cumsum of game_lengths}
```

優勢：降低梯度方差（中心極限定理：N 條軌跡的 gradient variance ≈ 1/N）。

---

## 11. 評估系統

### 11.1 離線評估（Offline Action Accuracy）

#### 方式 A：從預存 .npy 檔案

```bash
python evaluate.py --mode offline \
    --logits logits.npy \
    --targets targets.npy \
    --mask mask.npy \
    --top_k
```

#### 方式 B：從模型 + 驗證集推論

```bash
python evaluate.py --mode offline \
    --checkpoint checkpoints/bc_model/best_bc_model.pt \
    --data-path /data/converted_features_npy \
    --val-split 0.2 \
    --seed 42 \
    --batch-size 256
```

使用 `StreamingAccuracyTracker`：O(1) 記憶體複雜度，不保留任何大張量，只存純量 correct/total 計數器，適用於超大驗證集。

**輸出指標**：

```
類別準確率:
  總體準確率 (Overall) : XX.XX%
  dahai 準確率 : XX.XX%
  chow  準確率 : XX.XX%
  pong  準確率 : XX.XX%
  kong  準確率 : XX.XX%
  riichi準確率 : XX.XX%

局級指標（模型預測和了率）:
  總軌跡數                : N
  模型預測和了次數         : N
  模型預測和了率 (Pred Win): XX.XX%
  標註實際和了率 (True Win): XX.XX%
  預測和了精確率 (Precision): XX.XX%
  預測和了召回率 (Recall)   : XX.XX%
```

可選 Top-1/3/5 準確率。

### 11.2 自我對弈評估（Online Self-Play Statistics）

```bash
# 自我對弈（所有玩家用同一模型）
python evaluate.py --mode selfplay \
    --checkpoint checkpoints/ppo_lora/best_ppo_lora_model.pt \
    --num_games 1000 \
    --temperature 0 \
    --eval-mode attack

# PPO vs BC baseline 對比
python evaluate.py --mode selfplay \
    --checkpoint checkpoints/ppo_lora/best_ppo_lora_model.pt \
    --baseline-checkpoint checkpoints/bc_model/best_bc_model.pt \
    --num_games 1000

# PPO vs Mortal 對比（agent 對抗 3 個 Mortal AI）
python evaluate.py --mode selfplay \
    --checkpoint checkpoints/ppo_lora/best_ppo_lora_model.pt \
    --mortal-weights /path/to/mortal.pth \
    --num_games 1000
```

**輸出指標**（符合 Suphx 論文規範，per-hand 語義）：

```
============================================================
           麻將 AI 自我對弈統計報告
============================================================
  半莊數 (Games)        : 1000
  總局數 (Hands)         : 8234

  ── Per-Hand 指標（Suphx 學術標準）──
  和了率 (Win Rate)      : 24.31%  (2002/8234)
  放銃率 (Deal-in Rate)  : 12.15%  (1001/8234)
  流局率 (Draw Rate)     : 18.50%  (1523/8234)

  ── Game-Level 指標 ──
  平均終局分數           : 27234.5

  順位分佈:
    1 位 (一位): 32.40%  (324)
    2 位 (二位): 26.80%  (268)
    3 位 (三位): 22.10%  (221)
    4 位 (四位): 18.70%  (187)
============================================================
```

### 11.3 MahjongMetricTracker（`utli/evaluation_metrics.py`）

核心統計類別，支援：
- **Per-hand 語義**：和了率/放銃率的分母是 `total_hands`（每局牌數），而非 `total_games`（半莊數）
- **流局判定**：優先使用 proto 原生 `round_terminal.no_winner` 欄位，fallback 至分數閾值 ±4000
- **順位分佈**：以 game-level 統計 1/2/3/4 名百分比
- **report() / print_report()**：格式化輸出完整統計報告

---

## 12. 訓練與評估指令

### 12.1 完整訓練流程

```bash
# ===== 第一步：BC 預訓練（單頭） =====
python train_bc.py \
    --npz-file /data/converted_features_npy \
    --batch-size 256 \
    --num-epochs 100 \
    --learning-rate 1e-3 \
    --device cuda
# 輸出：checkpoints/bc_model/best_bc_model.pt

# ===== 第二步（可選）：多頭分權 + Focal Loss 微調 =====
python train_bc_multhead.py \
    --pretrained checkpoints/bc_model/best_bc_model.pt \
    --npz-file /data/converted_features_npy \
    --loss-mode focal \
    --focal-gamma 2.0 \
    --kong-weight 3.0 \
    --num-epochs 30
# 輸出：checkpoints/bc_multhead/best_multhead_model.pt

# ===== 第三步：PPO + LoRA 自我博弈（攻擊型，sparse reward） =====
python train_ppo.py \
    --bc-checkpoint checkpoints/bc_lora_focalloss/best_multhead_model.pt \
    --mode attack \
    --reward-mode sparse \
    --num-iterations 2000 \
    --learning-rate 5e-5 \
    --temperature 0.8 \
    --entropy-coef 0.01 \
    --num-trajectories 32
# 輸出：checkpoints/ppo_attack_bc_sparser_32batch/best_ppo_lora_model.pt

# ===== 第三步（可選）：PPO + LoRA 自我博弈（攻擊型，dense reward） =====
python train_ppo.py \
    --bc-checkpoint checkpoints/bc_lora_focalloss/best_multhead_model.pt \
    --mode attack \
    --reward-mode dense \
    --num-iterations 2000 \
    --learning-rate 5e-5 \
    --temperature 0.8 \
    --entropy-coef 0.01 \
    --num-trajectories 32

# ===== 第三步（可選）：PPO + LoRA 自我博弈（防守型，sparse reward） =====
python train_ppo.py \
    --bc-checkpoint checkpoints/bc_lora_focalloss/best_multhead_model.pt \
    --mode defense \
    --reward-mode sparse \
    --num-iterations 2000 \
    --learning-rate 5e-5 \
    --temperature 0.8 \
    --entropy-coef 0.01 \
    --num-trajectories 32
```

### 12.2 評估指令

```bash
# 離線 BC 準確率（從模型推論）
python evaluate.py --mode offline \
    --checkpoint checkpoints/bc_model/best_bc_model.pt \
    --data-path /data/converted_features_npy

# 離線 BC 準確率（預存 .npy）
python evaluate.py --mode offline \
    --logits bc_logits.npy \
    --targets bc_targets.npy \
    --top_k

# PPO 自我對弈評估
python evaluate.py --mode selfplay \
    --checkpoint checkpoints/ppo_lora_v4/best_ppo_lora_model.pt \
    --num_games 1000 \
    --temperature 0 \
    --output eval_report.txt

# PPO vs BC baseline
python evaluate.py --mode selfplay \
    --checkpoint checkpoints/ppo_lora_v4/best_ppo_lora_model.pt \
    --baseline-checkpoint checkpoints/bc_model/best_bc_model.pt \
    --num_games 1000

# PPO vs Mortal
python evaluate.py --mode selfplay \
    --checkpoint checkpoints/ppo_lora_v4/best_ppo_lora_model.pt \
    --mortal-weights /data/mortal_weights/mortal.pth \
    --num_games 1000
```

---

## 13. 專案目錄結構

```
/workspace/Mahjong/
├── README.md                        # 本文件
├── .gitignore
├── .gitmodules                      # mjx 子模組
│
├── model.py                         # DecisionMamba / DecisionMambaMultiHead / LoRALinear / MultiGrainedBlock
├── dataset.py                       # BehavioralCloningDataset / bc_collate_fn
│
├── train_bc.py                      # 第一階段：BC 預訓練主腳本
├── train_bc_multhead.py             # 第一階段延伸：多頭分權 + Focal Loss 微調
├── train_ppo.py                     # 第二階段：PPO + LoRA 自我博弈主腳本
├── evaluate.py                      # 評估主入口（offline + selfplay 雙模式）
│
├── utli/
│   ├── train_bc_step.py             # BC 單步訓練（CE/Focal + RTG + State 多任務損失）
│   ├── train_ppo_step.py            # PPO epoch 訓練（軌跡收集 + GAE + PPO Clip + Backward）
│   ├── runner.py                    # SelfPlayRunner（自我博弈環境 + 對手池 + 軌跡收集）
│   ├── rewards.py                   # MahjongRewardCalculator（五項 reward + 雙模式）
│   ├── lora_focal_finetune.py       # LoRA 喚醒 + Focal Loss 微調初始化
│   ├── focal_loss.py                # MahjongFocalLoss / build_default_alpha_weights
│   ├── evaluation_metrics.py        # 離線準確率 + MahjongMetricTracker + StreamingAccuracyTracker
│   └── mortal_agent.py              # Mortal AI 包裝器（mjx ↔ Mortal 橋接）
│
├── mjx/                             # 日麻遊戲引擎（C++ / pybind，子模組）
│   ├── mjx/                         # Python binding
│   │   ├── action.py
│   │   ├── observation.py
│   │   ├── state.py
│   │   └── ...
│   ├── include/mjx/                 # C++ 原始碼
│   │   ├── action.cpp               # 181 維動作編碼定義
│   │   ├── observation.cpp
│   │   ├── state.cpp
│   │   └── ...
│   └── tests_cpp/                   # C++ 單元測試
│
├── Mortal/                          # Mortal 模型（開源最強日麻 AI，子模組）
│   ├── mortal/
│   │   ├── model.py                 # Mortal Brain + DQN 架構
│   │   └── engine.py                # MortalEngine（react_batch 推理引擎）
│   └── libriichi/                   # Rust 麻將引擎（via maturin）
│
├── convert/                         # 資料轉換工具（mjai JSON → .npy 特徵）
├── checkpoints/                     # 模型 checkpoint 儲存目錄
│   ├── bc_model/                    # BC 預訓練權重
│   ├── bc_multhead/                 # 多頭分權微調權重
│   └── ppo_lora_v4/                 # PPO 訓練權重
│
├── log/                             # 訓練日誌
└── md/                              # 參考文檔
    ├── evaluate.md                  # 評估模組設計需求
    ├── REWARD_MODE_REFACTOR.md      # 雙模式攻守 reward 設計文檔
    └── feature_action_reference.md  # 特徵/動作空間參考手冊
```

---

## 14. 參考文獻

| 論文 / 技術 | 說明 |
|------------|------|
| **Suphx** (Microsoft Research Asia, 2019) | 麻將 AI 評估指標體系（和了率/放銃率/順位分佈） |
| **Decision Transformer** (Chen et al., 2021) | RTG-conditioned sequence modeling 範式 |
| **Mamba** (Gu & Dao, 2023) | State Space Model，線性時間複雜度的序列建模 |
| **LoRA** (Hu et al., 2021) | Low-Rank Adaptation，參數高效的微調技術 |
| **Focal Loss** (Lin et al., 2017) | 用於解決類別不平衡的損失函數 |
| **PPO** (Schulman et al., 2017) | Proximal Policy Optimization |
| **GAE** (Schulman et al., 2016) | Generalized Advantage Estimation |
| **Mortal** (Equim-chan, 2023) | 開源最強日麻 AI，作為對比基準 |

---

> 📝 **維護者**：kellen931214  
> 📅 **最後更新**：2026-06  
> 🔗 **GitHub**：[https://github.com/kellen931214/Mahjong](https://github.com/kellen931214/Mahjong)