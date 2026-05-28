import numpy as np
import os
import glob
from pathlib import Path
import gc
from tqdm import tqdm  # 📌 引入進度條模組

data_dir = "/workspace/Mahjong/converted_features"
output_dir = "/workspace/Mahjong/converted_features_npy"
os.makedirs(output_dir, exist_ok=True)

npz_files = sorted(glob.glob(os.path.join(data_dir, "converted_trajectories_chunk_*.npz")))

print(f"🚀 開始轉換資料格式 (安全防爆模式)...")

# ✨ 更好的解決方法：用 tqdm 包裹迴圈，並關閉原本的 print 避免洗版
for idx, npz_path in enumerate(tqdm(npz_files, desc="資料轉檔進度")):
    chunk_folder = Path(output_dir) / f"chunk_{idx}"
    chunk_folder.mkdir(parents=True, exist_ok=True)
    
    with np.load(npz_path, allow_pickle=False) as data:
        np.save(chunk_folder / "features.npy", data['features'])
        np.save(chunk_folder / "actions.npy", data['actions'])
        np.save(chunk_folder / "trajectory_boundaries.npy", data['trajectory_boundaries'])
        if 'rtgs' in data:
            np.save(chunk_folder / "rtgs.npy", data['rtgs'])
            
    del data
    gc.collect()

print(f"\n🎉 所有資料轉換完成！新路徑在: {output_dir}")