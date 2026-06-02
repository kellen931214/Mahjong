"""
將MJAI mjson格式轉換為README中定義的特徵格式
Spatial Channels (C=40) + Scalar Features (S=20) = 1380維特徵
動作空間 (Action Space) = 181維 (對接 mjx)
"""

import glob
import json
import gzip
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import IntEnum
import os

# ==================== Tile 定義 ====================

class TileType(IntEnum):
    """34種標準麻將牌"""
    M1 = 0; M2 = 1; M3 = 2; M4 = 3; M5 = 4; M6 = 5; M7 = 6; M8 = 7; M9 = 8
    P1 = 9; P2 = 10; P3 = 11; P4 = 12; P5 = 13; P6 = 14; P7 = 15; P8 = 16; P9 = 17
    S1 = 18; S2 = 19; S3 = 20; S4 = 21; S5 = 22; S6 = 23; S7 = 24; S8 = 25; S9 = 26
    EW = 27; SW = 28; WW = 29; NW = 30
    WD = 31; GD = 32; RD = 33  

TILE_STR_TO_TYPE = {
    "1m": 0, "2m": 1, "3m": 2, "4m": 3, "5m": 4, "6m": 5, "7m": 6, "8m": 7, "9m": 8,
    "1p": 9, "2p": 10, "3p": 11, "4p": 12, "5p": 13, "6p": 14, "7p": 15, "8p": 16, "9p": 17,
    "1s": 18, "2s": 19, "3s": 20, "4s": 21, "5s": 22, "6s": 23, "7s": 24, "8s": 25, "9s": 26,
    "E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33  
}

for k in list(TILE_STR_TO_TYPE.keys()):
    TILE_STR_TO_TYPE[k + "r"] = TILE_STR_TO_TYPE[k]

# ==================== MJX 動作編碼器 ====================

class MjxActionEncoder:
    @staticmethod
    def encode_discard(tile_type: int, is_red: bool = False) -> int:
        if is_red:
            if tile_type == 4: return 34
            if tile_type == 13: return 35
            if tile_type == 22: return 36
        return tile_type
    
    @staticmethod
    def encode_tsumogiri(tile_type: int, is_red: bool = False) -> int:
        if is_red:
            if tile_type == 4: return 71
            if tile_type == 13: return 72
            if tile_type == 22: return 73
        return tile_type + 37
    
    @staticmethod
    def encode_chi(base_tile: int, has_red: bool = False) -> int:
        if not has_red:
            if base_tile <= 8: return (base_tile % 9) + 74
            if base_tile <= 17: return (base_tile % 9) + 81
            return (base_tile % 9) + 88
        else:
            return {2: 95, 3: 96, 4: 97, 11: 98, 12: 99, 13: 100, 20: 101, 21: 102, 22: 103}.get(base_tile, 74)
    
    @staticmethod
    def encode_pon(tile_type: int, is_red: bool = False) -> int:
        if is_red:
            if tile_type == 4: return 138
            if tile_type == 13: return 139
            if tile_type == 22: return 140
        return tile_type + 104
    
    @staticmethod
    def encode_kan(tile_type: int) -> int:
        return tile_type + 141

# ==================== 特徵提取 ====================

@dataclass
class GameState:
    round_num: int
    honba: int
    kyotaku: int
    hands: List[List[int]]
    discards: List[List[int]]
    melds: List[List[List[int]]]
    scores: List[int]
    dora_indicators: List[int]
    dealer: int
    prevalent_wind: str
    reach_status: List[int] = field(default_factory=lambda: [0, 0, 0, 0]) # 修正: 加入立直狀態追蹤

class FeatureExtractor:
    def __init__(self, observer_idx: int):
        self.observer_idx = observer_idx
    
    def extract_spatial_features(self, state: GameState) -> np.ndarray:
        features = np.zeros((40, 34), dtype=np.uint8)
        
        hand_counts = [0] * 34
        for t in state.hands[self.observer_idx]:
            if 0 <= t < 34: hand_counts[t] += 1
        for t, count in enumerate(hand_counts):
            for i in range(min(count, 4)): features[i][t] = 1
        
        for meld_idx, meld in enumerate(state.melds[self.observer_idx][:4]):
            meld_counts = [0] * 34
            for t in meld: meld_counts[t] += 1
            for t, count in enumerate(meld_counts):
                if count > 0: features[4 + meld_idx][t] = 1
        
        for relative_pos in range(1, 4):
            abs_pos = (self.observer_idx + relative_pos) % 4
            for meld_idx, meld in enumerate(state.melds[abs_pos][:4]):
                meld_counts = [0] * 34
                for t in meld: meld_counts[t] += 1
                channel_idx = 8 + (relative_pos - 1) * 4 + meld_idx
                for t, count in enumerate(meld_counts):
                    if count > 0: features[channel_idx][t] = 1
        
        # 修正1: 牌河正確分配為 4 個通道 (range(4))，佔用 20-35
        for relative_pos in range(4):
            abs_pos = (self.observer_idx + relative_pos) % 4
            discards = state.discards[abs_pos]
            for group_idx in range(4):
                channel_idx = 20 + relative_pos * 4 + group_idx
                start_idx = group_idx * 6
                end_idx = min(start_idx + 6, len(discards))
                for discard_tile in discards[start_idx:end_idx]:
                    if 0 <= discard_tile < 34: features[channel_idx][discard_tile] = 1
        
        for dora_idx, dora in enumerate(state.dora_indicators[:4]):
            if 0 <= dora < 34: features[36 + dora_idx][dora] = 1
        
        return features
    
    def extract_scalar_features(self, state: GameState) -> np.ndarray:
        features = np.zeros(20, dtype=np.float32)
        for i in range(4):
            abs_pos = (self.observer_idx + i) % 4
            features[i] = max(0, (state.scores[abs_pos] + 50000) / 100000.0)
        
        used_tiles = sum(len(d) for d in state.discards)
        remaining_wall = 136 - used_tiles - sum(len(h) for h in state.hands)
        features[4] = max(0, remaining_wall / 70.0)
        
        if state.prevalent_wind == "E": features[5] = 1.0
        elif state.prevalent_wind == "S": features[6] = 1.0
        
        round_idx = state.round_num % 4
        if 0 <= round_idx < 4: features[7 + round_idx] = 1.0
        
        features[11] = min(state.honba / 30.0, 1.0)
        features[12] = min(state.kyotaku / 4.0, 1.0) # 修正: 讀取真實立直棒數量
        
        for i in range(4):
            abs_pos = (self.observer_idx + i) % 4
            features[13 + i] = float(state.reach_status[abs_pos]) # 修正: 讀取真實立直狀態
        
        relative_dealer = (state.dealer - self.observer_idx + 4) % 4
        if relative_dealer == 0: features[17] = 1.0
        elif relative_dealer == 1: features[18] = 1.0
        elif relative_dealer == 2: features[19] = 1.0
        
        return features
    
    def extract_features(self, state: GameState) -> np.ndarray:
        spatial_flat = self.extract_spatial_features(state).flatten()
        scalar = self.extract_scalar_features(state)
        return np.concatenate([spatial_flat, scalar])

# ==================== MJSON 解析與轉換 ====================

def parse_mjson_file(file_path: str) -> List[Dict]:
    events = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip(): events.append(json.loads(line.strip()))
    except: return []
    return events

def _encode_action_mjx(evt_type: str, evt: Dict) -> int:
    encoder = MjxActionEncoder()
    actor = evt.get('actor', -1)
    
    if evt_type == 'dahai':
        pai_str = evt.get('pai', '')
        tile_type = TILE_STR_TO_TYPE.get(pai_str, 0)
        is_red = 'r' in pai_str
        
        # 修正2: 精確判斷 tsumogiri 摸切
        if evt.get('tsumogiri', False):
            return encoder.encode_tsumogiri(tile_type, is_red)
        return encoder.encode_discard(tile_type, is_red)
    
    elif evt_type == 'chi':
        # 修正3: 使用 consumed 陣列精確還原吃牌
        pai = evt.get('pai', '')
        consumed = evt.get('consumed', [])
        all_tiles = [pai] + consumed
        tile_types = [TILE_STR_TO_TYPE.get(t, 0) for t in all_tiles]
        
        base_tile = min(tile_types) if tile_types else 0
        has_red = any('r' in t for t in all_tiles)
        return encoder.encode_chi(base_tile, has_red)
        
    elif evt_type == 'pon':
        pai = evt.get('pai', '')
        consumed = evt.get('consumed', [])
        all_tiles = [pai] + consumed
        tile_type = TILE_STR_TO_TYPE.get(pai, 0)
        has_red = any('r' in t for t in all_tiles)
        return encoder.encode_pon(tile_type, has_red)
        
    elif evt_type in ['kan', 'ankan', 'kakan']:
        pai = evt.get('pai', '')
        if not pai and evt.get('consumed'): pai = evt.get('consumed')[0] # 暗槓的防呆
        tile_type = TILE_STR_TO_TYPE.get(pai, 0)
        return encoder.encode_kan(tile_type)
        
    elif evt_type == 'hora':
        return 175 if evt.get('target') == actor else 176
    elif evt_type == 'reach' and evt.get('step') == 1: return 177
    elif evt_type == 'ryukyoku' and evt.get('reason') == 'kyushukyuhai': return 178
    elif evt_type == 'none': return 179
    return 180

def _update_game_state(state: GameState, evt: Dict) -> None:
    evt_type = evt.get('type')
    actor = evt.get('actor', -1)
    if actor < 0 or actor >= 4: return
    
    # 修正4: 補回摸牌(tsumo)與立直(reach)的狀態更新
    if evt_type == 'tsumo':
        tile_str = evt.get('pai')
        tile_type = TILE_STR_TO_TYPE.get(tile_str, -1)
        if tile_type != -1: state.hands[actor].append(tile_type)
            
    elif evt_type == 'reach' and evt.get('step') == 1:
        state.reach_status[actor] = 1
        state.kyotaku += 1
        
    elif evt_type == 'dahai':
        tile_str = evt.get('pai', '')
        tile_type = TILE_STR_TO_TYPE.get(tile_str, -1)
        if tile_type != -1:
            state.discards[actor].append(tile_type)
            if tile_type in state.hands[actor]: state.hands[actor].remove(tile_type)
            
    elif evt_type in ['chi', 'pon', 'kan', 'ankan', 'kakan']:
        pai = evt.get('pai', '')
        consumed = evt.get('consumed', [])
        all_tiles_str = [pai] + consumed if pai else consumed
        meld_tiles = [TILE_STR_TO_TYPE.get(t, -1) for t in all_tiles_str]
        meld_tiles = [t for t in meld_tiles if t != -1]
        
        if meld_tiles:
            state.melds[actor].append(meld_tiles)
            for t_str in consumed:
                t_type = TILE_STR_TO_TYPE.get(t_str, -1)
                # 確保只移除自己真實消耗掉的牌
                if t_type != -1 and t_type in state.hands[actor]: 
                    state.hands[actor].remove(t_type)

def extract_game_trajectories(events: List[Dict]) -> List[List[Dict]]:
    start_kyoku = next((e for e in events if e.get('type') == 'start_kyoku'), None)
    if not start_kyoku: return [[] for _ in range(4)]
    
    # 【新增】記錄起始分數，以便局終時計算分數差額 (Delta)
    start_scores = start_kyoku.get('scores', [25000] * 4)
    end_scores = start_scores.copy() 
    
    state = GameState(
        round_num=start_kyoku.get('kyoku', 0) - 1,
        honba=start_kyoku.get('honba', 0),
        kyotaku=start_kyoku.get('kyotaku', 0),
        hands=[[TILE_STR_TO_TYPE.get(t, -1) for t in start_kyoku.get('tehais', [[]])[i] if t in TILE_STR_TO_TYPE] for i in range(4)],
        discards=[[] for _ in range(4)], melds=[[] for _ in range(4)],
        scores=start_scores, # 使用剛才抓取的 start_scores
        dora_indicators=[TILE_STR_TO_TYPE.get(start_kyoku.get('dora_marker', ''), 0)],
        dealer=start_kyoku.get('oya', 0),
        prevalent_wind=start_kyoku.get('bakaze', 'E')
    )
    
    trajectories = [[] for _ in range(4)]
    decision_events = ['dahai', 'chi', 'pon', 'kan', 'ankan', 'kakan', 'reach', 'hora', 'none']
    
    for evt in events:
        evt_type = evt.get('type')
        actor = evt.get('actor', -1)
        
        # 【新增】擷取局終事件，更新最終分數
        if evt_type in ['hora', 'ryukyoku']:
            if 'scores' in evt:
                end_scores = evt['scores']
            elif 'deltas' in evt: # 防呆：有些 mjson 只有 deltas
                end_scores = [start_scores[i] + evt['deltas'][i] for i in range(4)]
        
        if evt_type in decision_events and 0 <= actor < 4:
            if evt_type == 'reach' and evt.get('step') != 1: pass
            else:
                try:
                    features = FeatureExtractor(actor).extract_features(state)
                    action_code = _encode_action_mjx(evt_type, evt)
                    trajectories[actor].append({'features': features, 'action': action_code})
                except Exception as e: pass
                
        _update_game_state(state, evt)
        
    # 【新增】迴圈結束後，計算每位玩家的真實 RTG，並賦予給軌跡中的每一個 step
    for p in range(4):
        # 計算真實分數差，並除以 10000 進行正規化
        score_delta = end_scores[p] - start_scores[p]
        rtg_value = score_delta / 10000.0  
        
        for step_data in trajectories[p]:
            step_data['rtg'] = rtg_value # 寫入 RTG
            
    return trajectories

def convert_mjson_directory(data_dir: str, output_dir: str, max_files: int = -1) -> None:
    os.makedirs(output_dir, exist_ok=True)
    all_mjson_files = sorted(glob.glob(os.path.join(data_dir, "*.mjson")))
    mjson_files = all_mjson_files[:max_files] if max_files > 0 else all_mjson_files
    print(f"Converting {len(mjson_files)}/{len(all_mjson_files)} mjson files...")
    
    # 【改進】採用 Chunking (分檔存儲) 徹底解決記憶體問題
    batch_size = 5000  # 每 5000 個檔案存成一個 chunk
    chunk_index = 0
    total_steps_all_chunks = 0
    
    for batch_start in range(0, len(mjson_files), batch_size):
        batch_end = min(batch_start + batch_size, len(mjson_files))
        batch_files = mjson_files[batch_start:batch_end]
        batch_trajectories = []
        
        print(f"\n[Chunk {chunk_index}] Processing files {batch_start + 1}-{batch_end}...")
        
        for file_idx, file_path in enumerate(batch_files, start=batch_start):
            basename = os.path.basename(file_path)
            print(f"  [{file_idx + 1}/{len(mjson_files)}] {basename}...", end=" ", flush=True)
            try:
                events = parse_mjson_file(file_path)
                if not events:
                    print("EMPTY")
                    continue
                trajectories = extract_game_trajectories(events)
                valid_trajectories = [t for t in trajectories if len(t) > 0]
                if valid_trajectories:
                    batch_trajectories.extend(valid_trajectories)
                    print(f"OK ({sum(len(t) for t in valid_trajectories)} steps)")
                else:
                    print(f"NO ACTIONS (traj lens: {[len(t) for t in trajectories]})")
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        # 直接輸出 dataset.py 可讀的 .npy 格式到 chunk 子目錄
        if batch_trajectories:
            features_list = [t['features'] for traj in batch_trajectories for t in traj]
            actions_list = [t['action'] for traj in batch_trajectories for t in traj]
            rtgs_list = [t['rtg'] for traj in batch_trajectories for t in traj]
            
            # 將目前這個 chunk 的軌跡長度轉換為邊界索引
            boundaries = np.cumsum([len(t) for t in batch_trajectories])
            
            # 建立 chunk 子目錄，直接寫入獨立 .npy 檔案（dataset.py 直接可讀）
            chunk_folder = os.path.join(output_dir, f"chunk_{chunk_index:03d}")
            os.makedirs(chunk_folder, exist_ok=True)
            
            np.save(os.path.join(chunk_folder, "features.npy"),
                    np.array(features_list, dtype=np.float32))
            np.save(os.path.join(chunk_folder, "actions.npy"),
                    np.array(actions_list, dtype=np.int64))
            np.save(os.path.join(chunk_folder, "rtgs.npy"),
                    np.array(rtgs_list, dtype=np.float32).reshape(-1, 1))
            np.save(os.path.join(chunk_folder, "trajectory_boundaries.npy"),
                    boundaries)
            
            steps_in_chunk = len(features_list)
            total_steps_all_chunks += steps_in_chunk
            
            print(f"  ✅ Chunk {chunk_index} saved to: {os.path.basename(chunk_folder)}/")
            print(f"     Steps in this chunk: {steps_in_chunk} | Total steps so far: {total_steps_all_chunks}")
            
            # 存完硬碟後，徹底刪除這些龐大的 List，讓 Python 進行 Garbage Collection
            del batch_trajectories, features_list, actions_list, rtgs_list
            chunk_index += 1
            
    print(f"\n🎉 All done! Processed {len(mjson_files)} files into {chunk_index} chunks.")
    print(f"Total steps generated across all chunks: {total_steps_all_chunks}")
    print(f"Output directory structure ready for dataset.py: {output_dir}/chunk_XXX/{{features,actions,rtgs,trajectory_boundaries}}.npy")

if __name__ == "__main__":
    import sys
    data_dir = "/workspace/Mahjong/data/mjai/2024"
    output_dir = "/data/converted_features_npy"
    max_files = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # -1 = all files, otherwise specify limit
    print(f"Output directory: {output_dir} (SSD)")
    convert_mjson_directory(data_dir, output_dir, max_files)
