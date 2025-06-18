#!/usr/bin/env python3
"""
NeMo-RL環境でのGPU間連携機能テスト
実際のワークロードでの性能検証
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    """簡単なテスト用モデル"""
    def __init__(self, input_size=1024, hidden_size=4096, output_size=1024):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

def create_synthetic_dataset(size=10000, input_dim=1024):
    """合成データセット作成"""
    X = torch.randn(size, input_dim)
    y = torch.randn(size, 1024)
    return torch.utils.data.TensorDataset(X, y)

def test_distributed_training(rank, world_size, model_size='medium'):
    """分散学習のテスト"""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # モデルサイズ設定
    if model_size == 'small':
        input_size, hidden_size, output_size = 512, 1024, 512
        batch_size = 64
    elif model_size == 'medium':
        input_size, hidden_size, output_size = 1024, 4096, 1024
        batch_size = 32
    else:  # large
        input_size, hidden_size, output_size = 2048, 8192, 2048
        batch_size = 16
    
    # モデル作成
    model = SimpleModel(input_size, hidden_size, output_size).cuda(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # データセット作成
    dataset = create_synthetic_dataset(1000, input_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # オプティマイザー
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 学習ループ
    ddp_model.train()
    total_time = 0
    comm_time = 0
    
    if rank == 0:
        print(f"分散学習テスト開始 (モデルサイズ: {model_size})")
        print(f"GPU数: {world_size}, バッチサイズ: {batch_size}")
    
    for epoch in range(3):
        epoch_start = time.time()
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.cuda(rank), target.cuda(rank)
            
            optimizer.zero_grad()
            
            # 通信時間測定開始
            comm_start = time.time()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 勾配同期 (DDP内で自動実行)
            optimizer.step()
            comm_time += time.time() - comm_start
            
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        
        if rank == 0:
            print(f"Epoch {epoch} 完了時間: {epoch_time:.4f}秒")
    
    if rank == 0:
        avg_epoch_time = total_time / 3
        comm_ratio = comm_time / total_time * 100
        print(f"平均エポック時間: {avg_epoch_time:.4f}秒")
        print(f"通信時間比率: {comm_ratio:.1f}%")
        
        # スケーリング効率の推定
        single_gpu_time_estimate = avg_epoch_time * world_size
        efficiency = single_gpu_time_estimate / avg_epoch_time / world_size * 100
        print(f"推定スケーリング効率: {efficiency:.1f}%")
    
    cleanup()

def test_memory_scaling(rank, world_size):
    """メモリ使用量のスケーリングテスト"""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("メモリスケーリングテスト開始")
    
    # 段階的にモデルサイズを増加
    model_sizes = [
        (1024, 2048, 1024),    # Small
        (2048, 4096, 2048),    # Medium
        (4096, 8192, 4096),    # Large
        (8192, 16384, 8192),   # XLarge
    ]
    
    for i, (input_size, hidden_size, output_size) in enumerate(model_sizes):
        try:
            torch.cuda.empty_cache()
            
            model = SimpleModel(input_size, hidden_size, output_size).cuda(rank)
            ddp_model = DDP(model, device_ids=[rank])
            
            # メモリ使用量確認
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(rank) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(rank) / 1024**3
                
                if rank == 0:
                    print(f"モデル{i+1}: メモリ使用量 {memory_allocated:.2f}GB / {memory_reserved:.2f}GB")
            
            # 簡単な前向き計算
            dummy_input = torch.randn(16, input_size).cuda(rank)
            with torch.no_grad():
                output = ddp_model(dummy_input)
            
            del model, ddp_model, dummy_input, output
            
        except RuntimeError as e:
            if rank == 0:
                print(f"モデル{i+1}: メモリ不足 - {str(e)}")
            break
    
    cleanup()

def benchmark_communication(rank, world_size):
    """通信性能のベンチマーク"""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print("通信性能ベンチマーク開始")
    
    # 各種サイズのテンソルでAllReduceテスト
    sizes_mb = [1, 10, 50, 100, 500]
    
    for size_mb in sizes_mb:
        size_elements = (size_mb * 1024 * 1024) // 4
        tensor = torch.randn(size_elements).cuda(rank)
        
        # ウォームアップ
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        torch.cuda.synchronize()
        
        # 性能測定
        iterations = 10
        start_time = time.time()
        
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        if rank == 0:
            avg_time = (end_time - start_time) / iterations
            # AllReduceの実効データ量
            effective_data = size_mb * 2 * (world_size - 1) / world_size
            bandwidth = effective_data / avg_time
            
            print(f"AllReduce {size_mb}MB: {avg_time:.4f}s, {bandwidth:.1f} MB/s")
    
    cleanup()

def main():
    if not torch.cuda.is_available():
        print("CUDAが利用できません。ホスト環境で実行してください。")
        return
    
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("マルチGPUテストには少なくとも2つのGPUが必要です")
        return
    
    print(f"NeMo-RL GPU連携テスト")
    print(f"GPU数: {world_size}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NCCL available: {dist.is_nccl_available()}")
    print("=" * 60)
    
    # テスト1: 通信性能ベンチマーク
    print("1. 通信性能ベンチマーク")
    mp.spawn(benchmark_communication,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    print("\n" + "=" * 60)
    
    # テスト2: 分散学習 (中サイズモデル)
    print("2. 分散学習性能テスト")
    mp.spawn(test_distributed_training,
             args=(world_size, 'medium'),
             nprocs=world_size,
             join=True)
    
    print("\n" + "=" * 60)
    
    # テスト3: メモリスケーリング
    print("3. メモリスケーリングテスト")
    mp.spawn(test_memory_scaling,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
    print("\n" + "=" * 60)
    print("全テスト完了")

if __name__ == "__main__":
    main() 