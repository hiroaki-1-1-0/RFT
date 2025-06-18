#!/usr/bin/env python3
"""
NCCL通信の帯域幅テスト専用スクリプト
マルチプロセスでNCCL AllReduceの性能を測定
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import argparse

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_bandwidth_test(rank, world_size, size_mb=100, iterations=10):
    setup(rank, world_size)
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # テストデータサイズ
    size_elements = (size_mb * 1024 * 1024) // 4  # float32
    
    # テンソル作成
    tensor = torch.randn(size_elements, device=device)
    
    # ウォームアップ
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    
    # 性能測定
    start_time = time.time()
    for _ in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    if rank == 0:
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        # AllReduceでは (world_size - 1) / world_size * 2 * size の通信が発生
        data_per_iteration = size_mb * 2 * (world_size - 1) / world_size
        bandwidth = data_per_iteration / avg_time
        
        print(f"データサイズ: {size_mb} MB")
        print(f"反復回数: {iterations}")
        print(f"総実行時間: {total_time:.4f}秒")
        print(f"平均実行時間: {avg_time:.4f}秒")
        print(f"推定帯域幅: {bandwidth:.1f} MB/s")
    
    cleanup()

def test_p2p_bandwidth(rank, world_size, size_mb=100, iterations=10):
    setup(rank, world_size)
    
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    size_elements = (size_mb * 1024 * 1024) // 4
    
    if rank == 0:
        # GPU 0からGPU 1へ送信
        if world_size > 1:
            tensor = torch.randn(size_elements, device=device)
            
            # ウォームアップ
            for _ in range(5):
                dist.send(tensor, dst=1)
            
            torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(iterations):
                dist.send(tensor, dst=1)
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / iterations
            bandwidth = size_mb / avg_time
            
            print(f"\nP2P通信テスト (GPU 0 -> GPU 1):")
            print(f"データサイズ: {size_mb} MB")
            print(f"平均転送時間: {avg_time:.4f}秒")
            print(f"帯域幅: {bandwidth:.1f} MB/s")
    
    elif rank == 1:
        # GPU 1でGPU 0からのデータを受信
        tensor = torch.zeros(size_elements, device=device)
        
        # ウォームアップ
        for _ in range(5):
            dist.recv(tensor, src=0)
        
        torch.cuda.synchronize()
        
        for _ in range(iterations):
            dist.recv(tensor, src=0)
        torch.cuda.synchronize()
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='NCCL帯域幅テスト')
    parser.add_argument('--size', type=int, default=100, help='テストデータサイズ (MB)')
    parser.add_argument('--iterations', type=int, default=10, help='反復回数')
    parser.add_argument('--test-type', choices=['allreduce', 'p2p', 'both'], 
                       default='both', help='テストタイプ')
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDAが利用できません")
        return
    
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("NCCLテストには少なくとも2つのGPUが必要です")
        return
    
    print(f"GPU数: {world_size}")
    print(f"NCCLバックエンド利用可能: {torch.distributed.is_nccl_available()}")
    
    if args.test_type in ['allreduce', 'both']:
        print("\n=== NCCL AllReduce帯域幅テスト ===")
        mp.spawn(run_bandwidth_test,
                args=(world_size, args.size, args.iterations),
                nprocs=world_size,
                join=True)
    
    if args.test_type in ['p2p', 'both']:
        print("\n=== NCCL P2P帯域幅テスト ===")
        mp.spawn(test_p2p_bandwidth,
                args=(world_size, args.size, args.iterations),
                nprocs=world_size,
                join=True)

if __name__ == "__main__":
    main() 