#!/bin/bash
# GPU間通信包括検証スクリプト (ホスト環境用)

echo "GPU間通信包括検証開始"
echo "================================"

# Phase 1: 基本環境確認
echo "Phase 1: 基本環境確認"
echo "nvidia-smi基本情報:"
nvidia-smi

echo -e "\nGPUトポロジー:"
nvidia-smi topo -m

echo -e "\nNVLink状態:"
nvidia-smi nvlink -s

echo -e "\nNVLink統計:"
nvidia-smi nvlink -g

# Phase 2: 性能測定
echo -e "\n\nPhase 2: 性能測定"
echo "CUDA samples テスト:"

if command -v bandwidthTest &> /dev/null; then
    echo "帯域幅テスト:"
    bandwidthTest
else
    echo "bandwidthTest not found"
fi

if command -v p2pBandwidthLatencyTest &> /dev/null; then
    echo "P2P帯域幅・レイテンシテスト:"
    p2pBandwidthLatencyTest
else
    echo "p2pBandwidthLatencyTest not found"
fi

# Phase 3: NCCLテスト
echo -e "\n\nPhase 3: NCCLテスト"
if command -v all_reduce_perf &> /dev/null; then
    echo "NCCL AllReduce性能テスト:"
    all_reduce_perf -b 1M -e 1G -f 2 -g 7
else
    echo "nccl-tests not found"
fi

echo "\n\n検証完了"
echo "================================"
