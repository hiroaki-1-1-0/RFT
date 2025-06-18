#!/usr/bin/env python3
"""
GPU間の連携機能を検証するためのテストスクリプト
"""

import torch
import torch.distributed as dist
import subprocess
import sys
import time
import numpy as np
from typing import List, Dict
import os

def check_cuda_availability():
    """CUDA環境の基本チェック"""
    print("=== CUDA環境の基本情報 ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1024**3:.1f} GB")
    print()

def check_gpu_topology():
    """nvidia-smiを使用してGPUトポロジーを確認"""
    print("=== GPU トポロジー ===")
    try:
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"nvidia-smi topo command failed: {e}")
    except FileNotFoundError:
        print("nvidia-smi not found")
    print()

def test_p2p_access():
    """GPU間のP2P (Peer-to-Peer) アクセスをテスト"""
    print("=== P2P アクセステスト ===")
    device_count = torch.cuda.device_count()
    
    if device_count < 2:
        print("P2Pテストには少なくとも2つのGPUが必要です")
        return
    
    p2p_matrix = []
    for i in range(device_count):
        row = []
        for j in range(device_count):
            if i == j:
                row.append("Self")
            else:
                torch.cuda.set_device(i)
                can_access = torch.cuda.can_device_access_peer(i, j)
                row.append("Yes" if can_access else "No")
        p2p_matrix.append(row)
    
    print("P2P Access Matrix (行: source GPU, 列: target GPU):")
    print("    " + "    ".join([f"GPU{j}" for j in range(device_count)]))
    for i, row in enumerate(p2p_matrix):
        print(f"GPU{i}: " + "  ".join([f"{cell:>4}" for cell in row]))
    print()

def test_memory_copy_performance():
    """GPU間メモリコピーの性能テスト"""
    print("=== GPU間メモリコピー性能テスト ===")
    device_count = torch.cuda.device_count()
    
    if device_count < 2:
        print("メモリコピーテストには少なくとも2つのGPUが必要です")
        return
    
    # テストデータサイズ (MB)
    test_sizes = [1, 10, 100, 500]  
    
    for size_mb in test_sizes:
        size_elements = (size_mb * 1024 * 1024) // 4  # float32
        print(f"\nテストサイズ: {size_mb} MB")
        
        for src_gpu in range(min(2, device_count)):  # 最初の2つのGPUでテスト
            for dst_gpu in range(min(2, device_count)):
                if src_gpu == dst_gpu:
                    continue
                
                torch.cuda.set_device(src_gpu)
                src_tensor = torch.randn(size_elements, device=f'cuda:{src_gpu}')
                
                # ウォームアップ
                for _ in range(3):
                    dst_tensor = src_tensor.to(f'cuda:{dst_gpu}')
                    torch.cuda.synchronize()
                
                # 性能測定
                torch.cuda.synchronize()
                start_time = time.time()
                dst_tensor = src_tensor.to(f'cuda:{dst_gpu}')
                torch.cuda.synchronize()
                end_time = time.time()
                
                transfer_time = end_time - start_time
                bandwidth = size_mb / transfer_time
                print(f"  GPU{src_gpu} -> GPU{dst_gpu}: {transfer_time:.4f}s, {bandwidth:.1f} MB/s")

def test_nccl_communication():
    """NCCL通信のテスト"""
    print("=== NCCL通信テスト ===")
    device_count = torch.cuda.device_count()
    
    if device_count < 2:
        print("NCCL通信テストには少なくとも2つのGPUが必要です")
        return
    
    # 環境変数設定
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    try:
        # プロセスグループ初期化
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', rank=0, world_size=1)
        
        # AllReduceテスト
        tensor_size = 1024 * 1024  # 1M elements
        tensors = []
        
        for gpu_id in range(device_count):
            torch.cuda.set_device(gpu_id)
            tensor = torch.ones(tensor_size, device=f'cuda:{gpu_id}') * (gpu_id + 1)
            tensors.append(tensor)
        
        print("AllReduceテスト前のテンソル値 (最初の5要素):")
        for i, tensor in enumerate(tensors):
            print(f"  GPU{i}: {tensor[:5].cpu().numpy()}")
        
        # AllReduce実行 (単一プロセスでは実際のAllReduceは実行されない)
        print("注意: 単一プロセス環境のため、実際のNCCL AllReduceは実行されません")
        
    except Exception as e:
        print(f"NCCL通信テストエラー: {e}")
    
    print()

def test_multi_gpu_computation():
    """マルチGPU計算のテスト"""
    print("=== マルチGPU計算テスト ===")
    device_count = torch.cuda.device_count()
    
    # 各GPUで並列計算
    results = []
    matrix_size = 2048
    
    print(f"各GPUで {matrix_size}x{matrix_size} 行列乗算を実行...")
    
    start_time = time.time()
    for gpu_id in range(device_count):
        torch.cuda.set_device(gpu_id)
        
        # 行列作成
        a = torch.randn(matrix_size, matrix_size, device=f'cuda:{gpu_id}')
        b = torch.randn(matrix_size, matrix_size, device=f'cuda:{gpu_id}')
        
        # 行列乗算
        c = torch.matmul(a, b)
        results.append(c)
    
    # 全GPU同期
    for gpu_id in range(device_count):
        torch.cuda.set_device(gpu_id)
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"全GPU計算完了時間: {total_time:.4f}秒")
    print(f"GPU使用率: 各GPUで{matrix_size}x{matrix_size}行列乗算が正常に完了")
    
    # 結果検証
    for i, result in enumerate(results):
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"警告: GPU{i}の計算結果に異常値が含まれています")
        else:
            print(f"GPU{i}: 計算結果正常 (平均値: {result.mean().item():.4f})")
    
    print()

def check_nvidia_ml():
    """nvidia-ml-pyを使用した詳細情報取得"""
    print("=== NVIDIA-ML 詳細情報 ===")
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"検出されたGPU数: {device_count}")
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # メモリ情報
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_total = mem_info.total / 1024**3
            mem_used = mem_info.used / 1024**3
            mem_free = mem_info.free / 1024**3
            
            # 温度
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # 使用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            print(f"GPU {i}: {name}")
            print(f"  メモリ: {mem_used:.1f}/{mem_total:.1f} GB ({mem_free:.1f} GB free)")
            print(f"  温度: {temp}°C")
            print(f"  GPU使用率: {util.gpu}%, メモリ使用率: {util.memory}%")
            
    except ImportError:
        print("pynvmlがインストールされていません")
        print("pip install nvidia-ml-py でインストールしてください")
    except Exception as e:
        print(f"NVIDIA-ML情報取得エラー: {e}")
    
    print()

def main():
    print("GPU間連携機能検証テスト開始\n")
    print("=" * 60)
    
    # 基本チェック
    check_cuda_availability()
    
    # トポロジーチェック
    check_gpu_topology()
    
    # P2Pアクセステスト
    test_p2p_access()
    
    # メモリコピー性能テスト
    test_memory_copy_performance()
    
    # NCCL通信テスト
    test_nccl_communication()
    
    # マルチGPU計算テスト
    test_multi_gpu_computation()
    
    # NVIDIA-ML詳細情報
    check_nvidia_ml()
    
    print("=" * 60)
    print("GPU間連携機能検証テスト完了")

if __name__ == "__main__":
    main() 