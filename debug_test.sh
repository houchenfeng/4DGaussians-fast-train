#!/bin/bash
# Debug测试 - 快速验证各模块功能
# 配置继承自 debug_test.py (少量迭代)

set -e

CONDA_ENV="4dgs"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

DATA_PATH="/home/lt/2024/data/N3D/multipleview/sear_steak"
GPU=1

echo "=========================================="
echo "Instant4D模块Debug测试"
echo "=========================================="
echo "配置基础: debug_test.py (少量迭代)"
echo "GPU: $GPU"
echo ""

# 测试1: Baseline
echo "[1/5] Baseline"
T1=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU python train.py -s "$DATA_PATH" --port 6101 \
    --expname "debug/baseline" --configs arguments/dynerf/debug_baseline.py \
    --quiet 2>&1 | tee /tmp/debug_baseline.log
T1_TIME=$(( $(date +%s) - T1 ))
P1=$(grep "Number of points" /tmp/debug_baseline.log | tail -1 | awk '{print $NF}')
echo "✓ ${T1_TIME}s, 点数: $P1"
echo ""

# 测试2: 网格剪枝
echo "[2/5] 网格剪枝"
T2=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU python train.py -s "$DATA_PATH" --port 6102 \
    --expname "debug/pruning" --configs arguments/dynerf/debug_pruning.py \
    --quiet 2>&1 | tee /tmp/debug_pruning.log
T2_TIME=$(( $(date +%s) - T2 ))
P2=$(grep "Number of points" /tmp/debug_pruning.log | tail -1 | awk '{print $NF}')
echo "✓ ${T2_TIME}s, 点数: $P2"
grep "\[Grid Pruning\]" /tmp/debug_pruning.log | head -3
echo ""

# 测试3: 各向同性
echo "[3/5] 各向同性"
T3=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU python train.py -s "$DATA_PATH" --port 6103 \
    --expname "debug/isotropic" --configs arguments/dynerf/debug_isotropic.py \
    --quiet 2>&1 | tee /tmp/debug_isotropic.log
T3_TIME=$(( $(date +%s) - T3 ))
P3=$(grep "Number of points" /tmp/debug_isotropic.log | tail -1 | awk '{print $NF}')
echo "✓ ${T3_TIME}s, 点数: $P3"
grep "\[Isotropic\]" /tmp/debug_isotropic.log | head -2
echo ""

# 测试4: 简化RGB
echo "[4/5] 简化RGB"
T4=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU python train.py -s "$DATA_PATH" --port 6104 \
    --expname "debug/simplified" --configs arguments/dynerf/debug_simplified.py \
    --quiet 2>&1 | tee /tmp/debug_simplified.log
T4_TIME=$(( $(date +%s) - T4 ))
P4=$(grep "Number of points" /tmp/debug_simplified.log | tail -1 | awk '{print $NF}')
echo "✓ ${T4_TIME}s, 点数: $P4"
grep "\[Simplified\]" /tmp/debug_simplified.log | head -2
echo ""

# 测试5: 全部改进
echo "[5/5] 全部改进"
T5=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU python train.py -s "$DATA_PATH" --port 6105 \
    --expname "debug/all" --configs arguments/dynerf/debug_all.py \
    --quiet 2>&1 | tee /tmp/debug_all.log
T5_TIME=$(( $(date +%s) - T5 ))
P5=$(grep "Number of points" /tmp/debug_all.log | tail -1 | awk '{print $NF}')
echo "✓ ${T5_TIME}s, 点数: $P5"
echo ""

# 结果汇总
echo "=========================================="
echo "Debug测试结果"
echo "=========================================="
printf "%-20s %10s %12s\n" "配置" "时间(秒)" "点数"
echo "------------------------------------------"
printf "%-20s %10s %12s\n" "Baseline" "$T1_TIME" "$P1"
printf "%-20s %10s %12s\n" "网格剪枝" "$T2_TIME" "$P2"
printf "%-20s %10s %12s\n" "各向同性" "$T3_TIME" "$P3"
printf "%-20s %10s %12s\n" "简化RGB" "$T4_TIME" "$P4"
printf "%-20s %10s %12s\n" "全部改进" "$T5_TIME" "$P5"
echo "=========================================="
echo ""
echo "输出: output/debug/{baseline,pruning,isotropic,simplified,all}/"
echo "日志: /tmp/debug_*.log"
