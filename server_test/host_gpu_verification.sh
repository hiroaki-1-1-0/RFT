#!/bin/bash
# ホスト環境でのGPU間連携検証スクリプト

echo "=== GPU基本情報 ==="
nvidia-smi

echo -e "\n=== GPUトポロジー ==="
nvidia-smi topo -m

echo -e "\n=== NVLink状態 ==="
nvidia-smi nvlink -s

echo -e "\n=== NVLink接続情報 ==="
nvidia-smi nvlink -g

echo -e "\n=== GPU詳細情報 ==="
for i in {0..6}; do
    echo "GPU $i:"
    nvidia-smi -i $i -q | grep -E "(Product Name|Memory|Temperature|Power|Performance)"
done

echo -e "\n=== CUDA samples テスト (存在する場合) ==="
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
