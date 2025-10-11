#!/bin/bash
# 完整训练测试 - 分模块执行，每次运行一个改进
# 用法: ./run_instant4d.sh [模块名]
# 模块名: baseline | pruning | isotropic | simplified | all

set -e

CONDA_ENV="4dgs"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

DATA_PATH="/home/lt/2024/data/N3D/multipleview/sear_steak"
GPU=1
ITER=14000

MODULE=${1:-all}  # 默认运行全部测试

echo "=========================================="
echo "Instant4D完整配置性能测试"
echo "=========================================="
echo "模块: $MODULE | 迭代: $ITER | GPU: $GPU"
echo "包含: 训练 + 渲染 + 指标计算"
echo ""

# 定义运行函数
run_test() {
    local name=$1
    local config=$2
    local port=$3
    local expname="full/$name"
    
    echo "=========================================="
    echo "测试: $name"
    echo "=========================================="
    echo "配置: $config"
    echo "输出: output/$expname/"
    echo ""
    
    # 训练
    echo "1. 训练中..."
    local start=$(date +%s)
    CUDA_VISIBLE_DEVICES=$GPU python train.py -s "$DATA_PATH" --port $port \
        --expname "$expname" --configs "$config" --save_iterations $ITER \
        2>&1 | tee "output/full_${name}.log"
    local train_time=$(( $(date +%s) - start ))
    
    local points=$(grep "Number of points" "output/full_${name}.log" | tail -1 | awk '{print $NF}')
    echo "  ✓ 训练完成: ${train_time}s ($(($train_time/60))分), 点数: $points"
    
    # 显示改进信息
    if [ "$name" != "baseline" ]; then
        grep -E "\[Grid Pruning\]|\[Isotropic\]|\[Simplified\]" "output/full_${name}.log" | head -5
    fi
    echo ""
    
    # 渲染
    echo "2. 渲染中..."
    python render.py -m "output/$expname" --iteration $ITER --quiet
    echo "  ✓ 渲染完成"
    echo ""
    
    # 计算指标
    echo "3. 计算指标..."
    python metrics.py -m "output/$expname" 2>&1 | tee "output/full_${name}_metrics.txt"
    local psnr=$(grep "PSNR" "output/full_${name}_metrics.txt" | head -1 | grep -o "[0-9.]*" | head -1)
    echo "  ✓ PSNR: $psnr"
    echo ""
    
    echo "结果:"
    echo "  训练时间: ${train_time}s ($(($train_time/60))分)"
    echo "  点数: $points"
    echo "  PSNR: $psnr"
    echo "  渲染: output/$expname/test/ours_$ITER/renders/"
    echo "  指标: output/full_${name}_metrics.txt"
    echo "=========================================="
    echo ""
}

# 根据参数运行对应测试
case $MODULE in
    baseline)
        run_test "baseline" "arguments/dynerf/sear_steak.py" 6201
        ;;
    
    pruning)
        run_test "pruning" "arguments/dynerf/sear_steak_pruning.py" 6202
        ;;
    
    isotropic)
        run_test "isotropic" "arguments/dynerf/sear_steak_isotropic.py" 6203
        ;;
    
    simplified)
        run_test "simplified" "arguments/dynerf/sear_steak_simplified.py" 6204
        ;;
    
    all)
        run_test "all" "arguments/dynerf/sear_steak_all.py" 6205
        ;;
    
    *)
        echo "用法: ./run.sh [模块名]"
        echo ""
        echo "可用模块:"
        echo "  baseline    - Baseline (无改进)"
        echo "  pruning     - 仅网格剪枝 (推荐)"
        echo "  isotropic   - 仅各向同性"
        echo "  simplified  - 仅简化RGB"
        echo "  all         - 全部改进"
        echo ""
        echo "示例:"
        echo "  ./run.sh baseline"
        echo "  ./run.sh pruning"
        echo "  ./run.sh all"
        exit 1
        ;;
esac

echo "✅ 测试完成！"
echo ""
echo "查看结果:"
echo "  cat output/full_${MODULE}_metrics.txt"
echo "  ls -lh output/full/${MODULE}/test/ours_${ITER}/renders/"
