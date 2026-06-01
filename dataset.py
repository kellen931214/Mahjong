import torch 
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence 
import numpy as np 
from typing import List, Dict 
import os 
from pathlib import Path
from collections import OrderedDict

class BehavioralCloningDataset(Dataset): 
    def __init__(self, data_path: str, min_trajectory_len: int = 1): 
        """ 
        支援包含 chunk_0, chunk_1... 子目錄的總目錄（內部為未壓縮的 .npy 檔案）
        """ 
        self.base_dir = Path(data_path)
        self.chunk_folders = sorted([d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
        
        if not self.chunk_folders: 
            raise FileNotFoundError(f"在 {data_path} 中找不到任何 chunk_* 子目錄！請確認轉檔腳本有正確執行。") 
        
        print(f"📂 找到 {len(self.chunk_folders)} 個真實 NPY Chunk 目錄") 
          
        self.trajectories = []  
        self._traj_total = 0 
        
        # ✨ 更好地解決方法 1：主進程只紀錄路徑字典，不直接調用 np.load 打開檔案描述符
        self.chunk_paths = {}
        # ✨ 更好地解決方法 2：將實體映射字典初始化為 None，留給子進程做延遲載入
        self.chunk_mmaps = None
          
        for chunk_idx, folder in enumerate(self.chunk_folders): 
            print(f"  快速掃描 chunk {chunk_idx}: {folder.name}...", end=' ', flush=True) 
              
            # 邊界資訊很小，直接讀入主進程記憶體中做軌跡分割計算
            boundaries = np.load(folder / 'trajectory_boundaries.npy') 
            
            # ✨ 只記錄各個資料矩陣的實體硬碟路徑
            self.chunk_paths[chunk_idx] = {
                'features': folder / "features.npy",
                'actions': folder / "actions.npy",
                'rtgs': folder / "rtgs.npy" if (folder / "rtgs.npy").exists() else None
            }
              
            prev_idx = 0 
            chunk_trajs = 0 
            for boundary in boundaries: 
                end_idx = int(boundary) 
                traj_len = end_idx - prev_idx 
                  
                if traj_len >= min_trajectory_len: 
                    self.trajectories.append({ 
                        'chunk_idx': chunk_idx, 
                        'start': prev_idx, 
                        'end': end_idx, 
                        'length': traj_len 
                    }) 
                    chunk_trajs += 1 
                  
                prev_idx = end_idx 
              
            print(f"✅ {chunk_trajs} 條軌跡") 
            self._traj_total += chunk_trajs 
        
        print(f"\n✅ 總共 {len(self.trajectories)} 條軌跡，跨 {len(self.chunk_folders)} 個 NPY Chunk 目錄\n") 

    def __len__(self) -> int: 
        return len(self.trajectories) 
      
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]: 
        """ 
        返回單條軌跡（子進程專屬延遲載入安全版）
        """ 
        traj_info = self.trajectories[idx] 
        chunk_idx = traj_info['chunk_idx']  
        start = traj_info['start'] 
        end = traj_info['end'] 
        length = traj_info['length'] 
        
        # 限制最大序列長度
        max_len = 512
        if length > max_len:
            start = end - max_len
            length = max_len
          
        # ✨ 延遲載入核心：每個 worker 進程第一次呼叫時建立該進程專屬的獨立 mmap
        # 使用 OrderedDict 實現 LRU 快取，容量設為 8（NVMe 無隨機讀取懲罰，cache 多可減少重開 mmap）
        if self.chunk_mmaps is None:
            self.chunk_mmaps = OrderedDict()
        
        if chunk_idx in self.chunk_mmaps:
            self.chunk_mmaps.move_to_end(chunk_idx)
        else:
            if len(self.chunk_mmaps) >= 8:
                self.chunk_mmaps.popitem(last=False)
            
            paths = self.chunk_paths[chunk_idx]
            self.chunk_mmaps[chunk_idx] = {
                'features': np.load(paths['features'], mmap_mode='r'),
                'actions': np.load(paths['actions'], mmap_mode='r'),
                'rtgs': np.load(paths['rtgs'], mmap_mode='r') if paths['rtgs'] is not None else None
            }
            
        mmaps = self.chunk_mmaps[chunk_idx]
          
        # 執行獨立進程空間內的零阻礙硬碟精準切片
        states = torch.from_numpy(mmaps['features'][start:end].copy()).float() 
        actions = torch.from_numpy(mmaps['actions'][start:end].copy()).long() 
        
        if mmaps['rtgs'] is not None:
            rtgs = torch.from_numpy(mmaps['rtgs'][start:end].copy()).float()
        else:
            rtgs = torch.ones((length, 1), dtype=torch.float32)
          
        timesteps = torch.arange(length, dtype=torch.long) 
        
        return { 
            'state': states, 
            'action': actions, 
            'rtg': rtgs,  
            'timesteps': timesteps, 
            'length': length 
        } 

def bc_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]: 
    """ 
    BC訓練的批次處理函數（自動補零對齊）
    """ 
    states_list = [item['state'] for item in batch] 
    actions_list = [item['action'] for item in batch] 
    rtgs_list = [item['rtg'] for item in batch] 
    timesteps_list = [item['timesteps'] for item in batch] 
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long) 
      
    state_padded = pad_sequence(states_list, batch_first=True, padding_value=0.0) 
    action_padded = pad_sequence(actions_list, batch_first=True, padding_value=-100) 
    tg_padded = pad_sequence(rtgs_list, batch_first=True, padding_value=0.0) 
    timesteps_padded = pad_sequence(timesteps_list, batch_first=True, padding_value=0) 
      
    max_len = state_padded.shape[1] 
    masks = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1) 
      
    return { 
        'state': state_padded,        
        'action': action_padded,      
        'rtg': tg_padded,            
        'timesteps': timesteps_padded,  
        'lengths': lengths,           
        'masks': masks.float()        
    }