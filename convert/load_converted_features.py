"""
加载和使用转换后的特征数据示例
"""

import numpy as np
import torch
from pathlib import Path

def load_converted_features(npz_file: str):
    """
    加载转换后的特征文件
    
    Args:
        npz_file: .npz文件路径
    
    Returns:
        features: (N, 1380) numpy数组
        actions: (N,) numpy数组
        trajectory_boundaries: 轨迹边界索引
    """
    data = np.load(npz_file)
    
    features = data['features']
    actions = data['actions']
    trajectory_boundaries = data['trajectory_boundaries']
    
    return features, actions, trajectory_boundaries


def parse_features(features):
    """
    解析特征为空间通道和标量特征
    
    Args:
        features: 单个特征向量 (1380,)
    
    Returns:
        spatial: (40, 34) 空间通道
        scalar: (20,) 标量特征
    """
    spatial = features[:1360].reshape(40, 34)
    scalar = features[1360:]
    return spatial, scalar


def create_feature_dataloader(features, actions, trajectory_boundaries, 
                              batch_size=32, shuffle=True):
    """
    创建PyTorch DataLoader
    
    Args:
        features: (N, 1380) 特征数组
        actions: (N,) 动作数组
        trajectory_boundaries: 轨迹边界
        batch_size: 批次大小
        shuffle: 是否打乱
    
    Returns:
        DataLoader对象
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    features_tensor = torch.from_numpy(features).float()
    actions_tensor = torch.from_numpy(actions).long()
    
    dataset = TensorDataset(features_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader


def print_feature_statistics(features, actions):
    """
    打印特征统计信息
    """
    print("=" * 60)
    print("特征统计信息")
    print("=" * 60)
    print(f"特征数量: {len(features)}")
    print(f"特征维度: {features.shape}")
    print(f"动作维度: {actions.shape}")
    print()
    
    print("空间通道 (0:1360)")
    spatial_features = features[:, :1360]
    print(f"  最小值: {spatial_features.min():.4f}")
    print(f"  最大值: {spatial_features.max():.4f}")
    print(f"  均值: {spatial_features.mean():.4f}")
    print(f"  非零元素: {(spatial_features > 0).sum()}")
    print()
    
    print("标量特征 (1360:1380)")
    scalar_features = features[:, 1360:]
    print(f"  最小值: {scalar_features.min():.4f}")
    print(f"  最大值: {scalar_features.max():.4f}")
    print(f"  均值: {scalar_features.mean():.4f}")
    print()
    
    print("动作分布")
    unique_actions, counts = np.unique(actions, return_counts=True)
    print(f"  不同动作数: {len(unique_actions)}")
    print(f"  最常见的5种动作:")
    for action, count in sorted(zip(unique_actions, counts), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    动作 {action}: {count} 次")


def main():
    # 加载转换后的数据
    npz_file = "/workspace/Mahjong/converted_features/converted_trajectories.npz"
    
    print(f"加载特征数据: {npz_file}")
    features, actions, trajectory_boundaries = load_converted_features(npz_file)
    
    # 打印统计信息
    print_feature_statistics(features, actions)
    print()
    
    # 示例：创建数据加载器
    print("创建 DataLoader...")
    dataloader = create_feature_dataloader(features, actions, trajectory_boundaries, batch_size=32)
    print(f"  批次数: {len(dataloader)}")
    print()
    
    # 示例：迭代一个批次
    print("示例：获取第一个批次")
    batch_features, batch_actions = next(iter(dataloader))
    print(f"  批次特征形状: {batch_features.shape}")
    print(f"  批次动作形状: {batch_actions.shape}")
    print(f"  批次动作: {batch_actions.numpy()}")
    print()
    
    # 示例：解析单个特征
    print("示例：解析第一个特征")
    spatial, scalar = parse_features(features[0])
    print(f"  空间通道形状: {spatial.shape}")
    print(f"  标量特征形状: {scalar.shape}")
    print(f"  标量特征: {scalar}")
    print()
    
    # 轨迹信息
    print("轨迹信息")
    print(f"  轨迹数: {len(trajectory_boundaries)}")
    print(f"  轨迹长度 (前5个):")
    prev_idx = 0
    for i, boundary in enumerate(trajectory_boundaries[:5]):
        traj_len = boundary - prev_idx
        print(f"    轨迹 {i}: {traj_len} 步")
        prev_idx = boundary


if __name__ == "__main__":
    main()
