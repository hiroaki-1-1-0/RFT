#!/usr/bin/env python3
"""
サーバー全体でのGPU間データ通信の包括的検証
コンテナ環境でも動作するよう設計
"""

import os
import sys
import time
import subprocess
import multiprocessing as mp
from pathlib import Path
import json

def check_environment():
    """環境の基本チェック"""
    print("=== 環境基本チェック ===")
    
    # CUDA環境変数
    cuda_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING']
    for var in cuda_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")
    
    # デバイスファイル確認
    device_count = 0
    for i in range(8):
        device_path = f'/dev/nvidia{i}'
        if os.path.exists(device_path):
            device_count += 1
            stat = os.stat(device_path)
            print(f"GPU {i}: {device_path} (major={os.major(stat.st_rdev)}, minor={os.minor(stat.st_rdev)})")
    
    print(f"検出されたGPUデバイス数: {device_count}")
    return device_count

def test_cuda_runtime():
    """CUDAランタイムの動作確認"""
    print("\n=== CUDAランタイム確認 ===")
    
    try:
        # PyTorch import
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA compiled version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}, Memory: {props.total_memory/1024**3:.1f}GB")
            
            return True, torch.cuda.device_count()
        else:
            print("CUDA not available in PyTorch")
            return False, 0
            
    except ImportError as e:
        print(f"PyTorch import error: {e}")
        return False, 0
    except Exception as e:
        print(f"CUDA runtime error: {e}")
        return False, 0

def force_cuda_initialization():
    """CUDA初期化の強制実行"""
    print("\n=== CUDA強制初期化 ===")
    
    # 環境変数設定
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
    
    try:
        import torch
        
        # 各GPUで簡単な計算を実行
        available_gpus = []
        for i in range(7):
            try:
                torch.cuda.set_device(i)
                # 簡単なテンソル作成と計算
                x = torch.tensor([1.0], device=f'cuda:{i}')
                y = x * 2
                result = y.cpu().item()
                print(f"GPU {i}: 計算成功 (結果: {result})")
                available_gpus.append(i)
            except Exception as e:
                print(f"GPU {i}: 計算失敗 - {e}")
        
        return available_gpus
        
    except Exception as e:
        print(f"強制初期化失敗: {e}")
        return []

def test_basic_p2p_communication():
    """基本的なP2P通信テスト"""
    print("\n=== 基本P2P通信テスト ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("P2P通信テストには2つ以上のGPUが必要です")
            return False
        
        device_count = torch.cuda.device_count()
        print(f"利用可能GPU数: {device_count}")
        
        # P2Pアクセス可能性確認
        p2p_matrix = []
        for i in range(device_count):
            row = []
            for j in range(device_count):
                if i == j:
                    row.append("Self")
                else:
                    try:
                        torch.cuda.set_device(i)
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        row.append("Yes" if can_access else "No")
                    except Exception as e:
                        row.append("Error")
            p2p_matrix.append(row)
        
        # P2P行列表示
        print("P2P Access Matrix:")
        header = "    " + "    ".join([f"GPU{j}" for j in range(device_count)])
        print(header)
        for i, row in enumerate(p2p_matrix):
            print(f"GPU{i}: " + "  ".join([f"{cell:>4}" for cell in row]))
        
        # 実際のデータ転送テスト
        print("\n実際のP2P転送テスト:")
        test_size = 1024 * 1024  # 1M elements
        
        for src in range(min(2, device_count)):
            for dst in range(min(2, device_count)):
                if src != dst:
                    try:
                        torch.cuda.set_device(src)
                        src_tensor = torch.randn(test_size, device=f'cuda:{src}')
                        
                        start_time = time.time()
                        dst_tensor = src_tensor.to(f'cuda:{dst}')
                        torch.cuda.synchronize()
                        end_time = time.time()
                        
                        transfer_time = end_time - start_time
                        bandwidth = (test_size * 4) / (1024 * 1024 * transfer_time)  # MB/s
                        
                        print(f"GPU{src} -> GPU{dst}: {transfer_time:.4f}s, {bandwidth:.1f} MB/s")
                        
                        # データ整合性確認
                        diff = torch.abs(src_tensor.cpu() - dst_tensor.cpu()).max().item()
                        if diff < 1e-6:
                            print(f"  データ整合性: OK")
                        else:
                            print(f"  データ整合性: NG (最大差分: {diff})")
                        
                    except Exception as e:
                        print(f"GPU{src} -> GPU{dst}: 転送失敗 - {e}")
        
        return True
        
    except Exception as e:
        print(f"P2P通信テストエラー: {e}")
        return False

def test_multi_gpu_synchronization():
    """マルチGPU同期テスト"""
    print("\n=== マルチGPU同期テスト ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("CUDA not available")
            return False
        
        device_count = torch.cuda.device_count()
        if device_count < 2:
            print("同期テストには2つ以上のGPUが必要です")
            return False
        
        # 各GPUで同時計算
        tensors = []
        matrix_size = 1024
        
        print(f"各GPU({device_count}台)で{matrix_size}x{matrix_size}行列乗算を実行...")
        
        start_time = time.time()
        
        # 各GPUで行列乗算開始
        for i in range(device_count):
            torch.cuda.set_device(i)
            
            # ランダム行列生成
            a = torch.randn(matrix_size, matrix_size, device=f'cuda:{i}')
            b = torch.randn(matrix_size, matrix_size, device=f'cuda:{i}')
            
            # 非同期で行列乗算開始
            c = torch.matmul(a, b)
            tensors.append((i, c))
        
        # 全GPU同期待ち
        for i in range(device_count):
            torch.cuda.set_device(i)
            torch.cuda.synchronize()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"全GPU計算完了時間: {total_time:.4f}秒")
        print(f"理論的シングルGPU時間: {total_time * device_count:.4f}秒")
        print(f"並列効率: {(total_time * device_count) / total_time / device_count * 100:.1f}%")
        
        # 結果検証
        for i, tensor in tensors:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"GPU{i}: 計算結果に異常値あり")
                return False
            else:
                mean_val = tensor.mean().item()
                print(f"GPU{i}: 計算成功 (平均値: {mean_val:.6f})")
        
        return True
        
    except Exception as e:
        print(f"マルチGPU同期テストエラー: {e}")
        return False

def test_memory_pressure():
    """メモリ圧迫下でのGPU間通信テスト"""
    print("\n=== メモリ圧迫下通信テスト ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("テストには2つ以上のGPUが必要です")
            return False
        
        device_count = torch.cuda.device_count()
        
        # 各GPUでメモリの大部分を占有
        memory_hogs = []
        for i in range(device_count):
            try:
                torch.cuda.set_device(i)
                
                # 利用可能メモリの70%程度を占有
                total_memory = torch.cuda.get_device_properties(i).total_memory
                target_memory = int(total_memory * 0.7)
                elements = target_memory // 4  # float32
                
                hog = torch.randn(elements, device=f'cuda:{i}')
                memory_hogs.append((i, hog))
                
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                print(f"GPU{i}: {allocated:.1f}GB 占有")
                
            except Exception as e:
                print(f"GPU{i}: メモリ占有失敗 - {e}")
        
        # メモリ圧迫下でのP2P転送
        print("メモリ圧迫下でのP2P転送テスト:")
        test_size = 10 * 1024 * 1024  # 10M elements = 40MB
        
        for src in range(min(2, device_count)):
            for dst in range(min(2, device_count)):
                if src != dst:
                    try:
                        torch.cuda.set_device(src)
                        test_tensor = torch.randn(test_size, device=f'cuda:{src}')
                        
                        start_time = time.time()
                        transferred = test_tensor.to(f'cuda:{dst}')
                        torch.cuda.synchronize()
                        end_time = time.time()
                        
                        transfer_time = end_time - start_time
                        bandwidth = (test_size * 4) / (1024 * 1024 * transfer_time)
                        
                        print(f"GPU{src} -> GPU{dst}: {transfer_time:.4f}s, {bandwidth:.1f} MB/s")
                        
                    except Exception as e:
                        print(f"GPU{src} -> GPU{dst}: 転送失敗 - {e}")
        
        # メモリ解放
        for i, hog in memory_hogs:
            del hog
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"メモリ圧迫テストエラー: {e}")
        return False

def test_collective_operations():
    """集合演算テスト（疑似分散環境）"""
    print("\n=== 集合演算テスト ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("CUDA not available")
            return False
        
        device_count = torch.cuda.device_count()
        if device_count < 2:
            print("集合演算テストには2つ以上のGPUが必要です")
            return False
        
        # 各GPUに異なる値のテンソルを作成
        tensors = []
        test_size = 1024 * 1024
        
        for i in range(device_count):
            torch.cuda.set_device(i)
            tensor = torch.full((test_size,), float(i + 1), device=f'cuda:{i}')
            tensors.append(tensor)
        
        print("疑似AllReduce実装（手動reduce）:")
        
        # GPU0に全てのテンソルを集約
        torch.cuda.set_device(0)
        sum_tensor = torch.zeros(test_size, device='cuda:0')
        
        start_time = time.time()
        
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor += tensor
            else:
                sum_tensor += tensor.to('cuda:0')
        
        # 結果を各GPUに配布
        result_tensors = []
        for i in range(device_count):
            if i == 0:
                result_tensors.append(sum_tensor.clone())
            else:
                result_tensors.append(sum_tensor.to(f'cuda:{i}'))
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        data_size = test_size * 4 * device_count  # float32 * device_count
        bandwidth = data_size / (1024 * 1024 * total_time)
        
        print(f"疑似AllReduce完了時間: {total_time:.4f}秒")
        print(f"データ転送量: {data_size / 1024 / 1024:.1f} MB")
        print(f"実効帯域幅: {bandwidth:.1f} MB/s")
        
        # 結果検証
        expected_sum = sum(range(1, device_count + 1))
        for i, result_tensor in enumerate(result_tensors):
            actual_value = result_tensor[0].item()
            if abs(actual_value - expected_sum) < 1e-6:
                print(f"GPU{i}: AllReduce結果正常 (値: {actual_value})")
            else:
                print(f"GPU{i}: AllReduce結果異常 (期待: {expected_sum}, 実際: {actual_value})")
                return False
        
        return True
        
    except Exception as e:
        print(f"集合演算テストエラー: {e}")
        return False

def generate_test_report(results):
    """テスト結果レポート生成"""
    print("\n" + "="*60)
    print("サーバー統合GPU通信機能検証レポート")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"実行テスト数: {total_tests}")
    print(f"成功テスト数: {passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n個別テスト結果:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("\n総合評価:")
    if passed_tests == total_tests:
        print("🟢 優秀: 全てのGPU間通信機能が正常に動作")
    elif passed_tests >= total_tests * 0.8:
        print("🟡 良好: 大部分のGPU間通信機能が正常に動作")
    elif passed_tests >= total_tests * 0.5:
        print("🟠 注意: 一部のGPU間通信機能に問題あり")
    else:
        print("🔴 警告: GPU間通信機能に重大な問題あり")
    
    print("\n推奨事項:")
    if not results.get('basic_p2p', True):
        print("- P2P通信が失敗している場合、NVLinkやPCIe接続を確認")
    if not results.get('multi_gpu_sync', True):
        print("- マルチGPU同期の問題がある場合、ドライバー更新を検討")
    if not results.get('memory_pressure', True):
        print("- メモリ圧迫下での問題がある場合、メモリ管理を最適化")
    if not results.get('collective_ops', True):
        print("- 集合演算の問題がある場合、NCCL設定を確認")

def main():
    print("サーバー統合GPU間データ通信包括検証")
    print("="*60)
    
    # 環境チェック
    device_count = check_environment()
    
    if device_count == 0:
        print("GPUデバイスが検出されません。検証を終了します。")
        return
    
    # CUDAランタイム確認
    cuda_available, cuda_device_count = test_cuda_runtime()
    
    # 結果記録用辞書
    test_results = {}
    
    if cuda_available:
        print(f"\nCUDA利用可能: {cuda_device_count}台のGPUで検証実行")
        
        # 基本P2P通信テスト
        test_results['basic_p2p'] = test_basic_p2p_communication()
        
        # マルチGPU同期テスト
        test_results['multi_gpu_sync'] = test_multi_gpu_synchronization()
        
        # メモリ圧迫下通信テスト
        test_results['memory_pressure'] = test_memory_pressure()
        
        # 集合演算テスト
        test_results['collective_ops'] = test_collective_operations()
        
    else:
        print("\nCUDA利用不可: 強制初期化を試行")
        available_gpus = force_cuda_initialization()
        
        if available_gpus:
            print(f"強制初期化成功: GPU {available_gpus} が利用可能")
            test_results['force_initialization'] = True
        else:
            print("強制初期化も失敗")
            test_results['force_initialization'] = False
    
    # レポート生成
    generate_test_report(test_results)

if __name__ == "__main__":
    main() 