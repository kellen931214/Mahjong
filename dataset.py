import torch 
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence 
import numpy as np 
from typing import List, Dict 
import os 
from pathlib import Path

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
        
        # ✨ 更好的解決方法：預先建立硬碟映射指標字典，避免 __getitem__ 重複打開檔案
        self.chunk_mmaps = {}
         
        for chunk_idx, folder in enumerate(self.chunk_folders): 
            print(f"  快速映射 chunk {chunk_idx}: {folder.name}...", end=' ', flush=True) 
             
            # 建立極輕量、不吃記憶體的硬碟核心映射
            boundaries = np.load(folder / 'trajectory_boundaries.npy', mmap_mode='r') 
            features_mmap = np.load(folder / "features.npy", mmap_mode='r')
            actions_mmap = np.load(folder / "actions.npy", mmap_mode='r')
            
            rtg_path = folder / "rtgs.npy"
            rtgs_mmap = np.load(rtg_path, mmap_mode='r') if rtg_path.exists() else None
            
            # 快取這些指標（這只是虛擬指標，佔用 RAM 微乎其微）
            self.chunk_mmaps[chunk_idx] = {
                'features': features_mmap,
                'actions': actions_mmap,
                'rtgs': rtgs_mmap
            }
             
            prev_idx = 0 
            chunk_trajs = 0 
            for boundary in boundaries: 
                end_idx = int(boundary) 
                traj_len = end_idx - prev_idx 
                 
                if traj_len >= min_trajectory_len: 
                    self.trajectories.append({ 
                        'chunk_idx': chunk_idx, # 改存索引，方便直接查表
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
        返回單條軌跡 
        """ 
        traj_info = self.trajectories[idx] 
        chunk_idx = traj_info['chunk_idx']  
        start = traj_info['start'] 
        end = traj_info['end'] 
        length = traj_info['length'] 
        
        # 📌 研究優化防線：麻將特徵維度高（1380），強行把最大長度壓在 512 步內，防止 pad_sequence 記憶體大爆炸
        max_len = 512
        if length > max_len:
            start = end - max_len
            length = max_len
         
        # ✨ 真正的零延遲速度：直接從預先映射好的指標做硬碟精準切片
        mmaps = self.chunk_mmaps[chunk_idx]
         
        # 精準切片取出這一段，並使用 .copy() 拉進實體記憶體
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
            'rtg': rtgs,  # 📌 修正原本的語法筆誤
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
    rtg_padded = pad_sequence(rtgs_list, batch_first=True, padding_value=0.0) 
    timesteps_padded = pad_sequence(timesteps_list, batch_first=True, padding_value=0) 
     
    max_len = state_padded.shape[1] 
    masks = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1) 
     
    return { 
        'state': state_padded,        
        'action': action_padded,      
        'rtg': rtg_padded,            
        'timesteps': timesteps_padded,  
        'lengths': lengths,           
        'masks': masks.float()        
    }