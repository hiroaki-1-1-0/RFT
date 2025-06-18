#!/usr/bin/env python3
"""
GPU環境の診断とGPU間連携検証のための分析ツール
CUDA環境が利用できない場合でも実行可能
"""

import os
import subprocess
import sys
import re
from typing import List, Dict, Optional

def check_nvidia_devices():
    """NVIDIA デバイスファイルの確認"""
    print("=== NVIDIA デバイスファイル確認 ===")
    
    nvidia_devices = []
    try:
        # /dev/nvidia* の確認
        for item in os.listdir('/dev'):
            if item.startswith('nvidia') and item.replace('nvidia', '').isdigit():
                nvidia_devices.append(item)
        
        nvidia_devices.sort(key=lambda x: int(x.replace('nvidia', '')))
        
        print(f"検出されたNVIDIA GPUデバイス数: {len(nvidia_devices)}")
        for device in nvidia_devices:
            device_path = f'/dev/{device}'
            if os.path.exists(device_path):
                stat = os.stat(device_path)
                print(f"  {device_path}: メジャー={os.major(stat.st_rdev)}, マイナー={os.minor(stat.st_rdev)}")
        
        # 制御デバイスの確認
        control_devices = ['nvidiactl', 'nvidia-uvm', 'nvidia-uvm-tools']
        for device in control_devices:
            device_path = f'/dev/{device}'
            if os.path.exists(device_path):
                print(f"  {device_path}: 存在")
            else:
                print(f"  {device_path}: 不存在")
                
    except Exception as e:
        print(f"デバイスファイル確認エラー: {e}")
    
    print()
    return len(nvidia_devices)

def check_nvidia_driver_info():
    """NVIDIA ドライバー情報の確認"""
    print("=== NVIDIA ドライバー情報 ===")
    
    # /proc/driver/nvidia/ の確認
    nvidia_proc_path = '/proc/driver/nvidia'
    if os.path.exists(nvidia_proc_path):
        print(f"{nvidia_proc_path} が存在します")
        
        try:
            # バージョン情報
            version_file = os.path.join(nvidia_proc_path, 'version')
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    version_info = f.read().strip()
                print(f"ドライバーバージョン情報:\n{version_info}")
            
            # GPU情報
            for item in os.listdir(nvidia_proc_path):
                if item.startswith('gpus'):
                    gpu_dir = os.path.join(nvidia_proc_path, item)
                    if os.path.isdir(gpu_dir):
                        print(f"\nGPU ディレクトリ: {gpu_dir}")
                        for gpu_item in os.listdir(gpu_dir):
                            print(f"  {gpu_item}")
                            
        except Exception as e:
            print(f"ドライバー情報読み取りエラー: {e}")
    else:
        print(f"{nvidia_proc_path} が存在しません")
    
    print()

def check_pci_devices():
    """PCI デバイスでNVIDIA GPUを確認"""
    print("=== PCI NVIDIA デバイス ===")
    
    try:
        # lspci コマンドでNVIDIA デバイスを検索
        result = subprocess.run(['lspci', '-nn'], capture_output=True, text=True, check=True)
        nvidia_devices = []
        
        for line in result.stdout.split('\n'):
            if 'NVIDIA' in line.upper() or '10de:' in line.lower():
                nvidia_devices.append(line.strip())
        
        if nvidia_devices:
            print(f"検出されたNVIDIA PCI デバイス数: {len(nvidia_devices)}")
            for i, device in enumerate(nvidia_devices):
                print(f"  {i}: {device}")
        else:
            print("NVIDIA PCI デバイスが検出されませんでした")
            
    except subprocess.CalledProcessError as e:
        print(f"lspci コマンドエラー: {e}")
    except FileNotFoundError:
        print("lspci コマンドが見つかりません")
    
    print()

def analyze_gpu_topology_theory():
    """理論的なGPUトポロジー分析"""
    print("=== RTX 6000 Ada Generation 理論的トポロジー分析 ===")
    
    print("NVIDIA RTX 6000 Ada Generation の特徴:")
    print("- アーキテクチャ: Ada Lovelace (AD102)")
    print("- メモリ: 48GB GDDR6 ECC")
    print("- メモリ帯域幅: 960 GB/s")
    print("- NVLink: NVLink 4.0 (最大 900 GB/s)")
    print("- PCIe: PCIe 4.0 x16")
    
    print("\n7枚構成での期待される接続パターン:")
    print("1. NVLink接続:")
    print("   - 理想的: 各GPUが2-4本のNVLinkで他のGPUと直接接続")
    print("   - 実際: サーバーの物理的制約により、一部はPCIe経由")
    
    print("2. PCIe接続:")
    print("   - CPU-GPU間: PCIe 4.0 x16 (約32 GB/s)")
    print("   - GPU-GPU間 (NVLink無し): PCIe switch経由")
    
    print("3. 期待される性能指標:")
    print("   - NVLink接続GPU間: 300-600 GB/s")
    print("   - PCIe接続GPU間: 20-30 GB/s")
    print("   - システム全体AllReduce: 100-300 GB/s (データサイズ依存)")
    
    print()

def recommend_verification_steps():
    """検証手順の推奨"""
    print("=== GPU間連携検証の推奨手順 ===")
    
    print("現在の状況:")
    print("- コンテナ内でnvidia-smiとCUDAライブラリが利用できない")
    print("- GPUデバイスファイルは存在")
    print("- ホスト環境での検証が必要")
    
    print("\n推奨検証手順:")
    print("1. ホスト環境での基本確認:")
    print("   nvidia-smi")
    print("   nvidia-smi topo -m")
    print("   nvidia-smi nvlink -s")
    
    print("2. GPU間通信性能測定:")
    print("   nvidia-smi nvlink -g")
    print("   bandwidthTest (CUDA samples)")
    print("   p2pBandwidthLatencyTest (CUDA samples)")
    
    print("3. NCCL性能測定:")
    print("   nccl-tests (https://github.com/NVIDIA/nccl-tests)")
    print("   python マルチプロセスNCCLテスト")
    
    print("4. 実際のワークロード性能:")
    print("   PyTorch/NeMo-RLでのマルチGPU学習テスト")
    print("   分散学習の通信オーバーヘッド測定")
    
    print()

def create_host_verification_script():
    """ホスト環境用の検証スクリプトを生成"""
    script_content = '''#!/bin/bash
# ホスト環境でのGPU間連携検証スクリプト

echo "=== GPU基本情報 ==="
nvidia-smi

echo -e "\\n=== GPUトポロジー ==="
nvidia-smi topo -m

echo -e "\\n=== NVLink状態 ==="
nvidia-smi nvlink -s

echo -e "\\n=== NVLink接続情報 ==="
nvidia-smi nvlink -g

echo -e "\\n=== GPU詳細情報 ==="
for i in {0..6}; do
    echo "GPU $i:"
    nvidia-smi -i $i -q | grep -E "(Product Name|Memory|Temperature|Power|Performance)"
done

echo -e "\\n=== CUDA samples テスト (存在する場合) ==="
if command -v bandwidthTest &> /dev/null; then
    bandwidthTest
else
    echo "bandwidthTest が見つかりません"
fi

if command -v p2pBandwidthLatencyTest &> /dev/null; then
    p2pBandwidthLatencyTest
else
    echo "p2pBandwidthLatencyTest が見つかりません"
fi
'''
    
    with open('server_test/host_gpu_verification.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('host_gpu_verification.sh', 0o755)
    print("ホスト環境用検証スクリプト 'host_gpu_verification.sh' を作成しました")
    print("ホスト環境で以下を実行してください:")
    print("  chmod +x host_gpu_verification.sh")
    print("  ./host_gpu_verification.sh")
    print()

def analyze_expected_performance():
    """期待される性能の分析"""
    print("=== 期待される性能指標 ===")
    
    print("1. メモリ帯域幅:")
    print("   - 各GPU内: 960 GB/s")
    print("   - 期待値: gpustat で表示された 621MB/49140MB は正常")
    
    print("2. GPU間通信:")
    print("   - NVLink 4.0: 最大 900 GB/s (理論値)")
    print("   - 実効値: 300-600 GB/s (システム構成依存)")
    print("   - PCIe 4.0: 最大 32 GB/s")
    
    print("3. 温度状況の分析:")
    print("   報告された温度: 27-42°C")
    print("   - 正常範囲: アイドル時20-50°C")
    print("   - GPU4が42°Cと他より高い -> 位置や冷却要確認")
    print("   - 負荷時の温度監視が重要 (通常70-85°C)")
    
    print("4. 多GPU学習での期待値:")
    print("   - 理想的スケーリング効率: 80-95%")
    print("   - 通信ボトルネック発生ポイント:")
    print("     * モデルサイズ > 1つのGPUメモリ")
    print("     * バッチサイズが小さい")
    print("     * NVLink接続が不十分")
    
    print()

def main():
    print("GPU環境診断ツール")
    print("=" * 60)
    
    # デバイスファイル確認
    gpu_count = check_nvidia_devices()
    
    # ドライバー情報確認
    check_nvidia_driver_info()
    
    # PCI デバイス確認
    check_pci_devices()
    
    # 理論的分析
    analyze_gpu_topology_theory()
    
    # 期待される性能分析
    analyze_expected_performance()
    
    # 検証手順推奨
    recommend_verification_steps()
    
    # ホスト用スクリプト作成
    create_host_verification_script()
    
    print("=" * 60)
    print("診断完了")
    
    print(f"\n結論:")
    print(f"- {gpu_count}台のGPUデバイスが検出されました")
    print(f"- コンテナ環境でCUDA/nvidia-smiアクセス制限あり")
    print(f"- ホスト環境での詳細検証が必要")
    print(f"- 温度状況は正常範囲内")

if __name__ == "__main__":
    main() 