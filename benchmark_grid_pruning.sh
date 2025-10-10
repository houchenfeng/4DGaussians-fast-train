#!/bin/bash

# 网格剪枝性能对比脚本
# 比较有无 Grid Pruning 的训练时间差异

echo "=========================================="
echo "Instant4D 网格剪枝性能对比测试"
echo "=========================================="

DATA_PATH="/home/lt/2024/data/N3D/multipleview/sear_steak"
CONFIG_PATH="arguments/dynerf/sear_steak.py"
PORT=6017
IP="127.0.0.4"

# 检查数据路径
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据路径不存在: $DATA_PATH"
    exit 1
fi

# 测试1: 不使用网格剪枝
echo ""
echo "=========================================="
echo "测试1: 训练不使用网格剪枝 (原始 4DGaussians)"
echo "=========================================="

# 临时禁用网格剪枝
sed -i 's/use_grid_pruning=True/use_grid_pruning=False/g' $CONFIG_PATH

echo "开始训练 (不使用网格剪枝)..."
echo "输出日志: training_without_pruning.log"

START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python train.py \
    -s "$DATA_PATH" \
    --port $PORT \
    --ip $IP \
    --expname "dynerf/sear_steak_without_pruning" \
    --configs $CONFIG_PATH \
    --quiet > training_without_pruning.log 2>&1

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION_WITHOUT=$((END_TIME - START_TIME))

if [ $EXIT_CODE -ne 0 ]; then
    echo "警告: 训练出现错误 (退出码: $EXIT_CODE)"
    echo "查看日志: training_without_pruning.log"
else
    echo "✓ 训练完成 (不使用网格剪枝)"
fi

# 提取关键指标
echo ""
echo "提取训练指标..."
POINTS_WITHOUT=$(grep "Number of points at initialisation" training_without_pruning.log | tail -1 | awk '{print $NF}')
echo "  初始点云数量: ${POINTS_WITHOUT:-未知}"
echo "  训练时间: ${DURATION_WITHOUT} 秒 ($(($DURATION_WITHOUT / 60)) 分钟)"

# 测试2: 使用网格剪枝
echo ""
echo "=========================================="
echo "测试2: 训练使用网格剪枝 (Instant4D 加速)"
echo "=========================================="

# 启用网格剪枝
sed -i 's/use_grid_pruning=False/use_grid_pruning=True/g' $CONFIG_PATH

echo "开始训练 (使用网格剪枝)..."
echo "输出日志: training_with_pruning.log"

START_TIME=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python train.py \
    -s "$DATA_PATH" \
    --port $PORT \
    --ip $IP \
    --expname "dynerf/sear_steak_with_pruning" \
    --configs $CONFIG_PATH \
    --quiet > training_with_pruning.log 2>&1

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION_WITH=$((END_TIME - START_TIME))

if [ $EXIT_CODE -ne 0 ]; then
    echo "警告: 训练出现错误 (退出码: $EXIT_CODE)"
    echo "查看日志: training_with_pruning.log"
else
    echo "✓ 训练完成 (使用网格剪枝)"
fi

# 提取关键指标
echo ""
echo "提取训练指标..."
POINTS_WITH=$(grep "Number of points at initialisation" training_with_pruning.log | tail -1 | awk '{print $NF}')
PRUNING_INFO=$(grep "\[Grid Pruning\]" training_with_pruning.log | tail -3)
echo "  网格剪枝信息:"
echo "$PRUNING_INFO" | sed 's/^/    /'
echo "  初始点云数量: ${POINTS_WITH:-未知}"
echo "  训练时间: ${DURATION_WITH} 秒 ($(($DURATION_WITH / 60)) 分钟)"

# 性能对比
echo ""
echo "=========================================="
echo "性能对比结果"
echo "=========================================="

if [ -n "$DURATION_WITHOUT" ] && [ -n "$DURATION_WITH" ] && [ $DURATION_WITHOUT -gt 0 ]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $DURATION_WITHOUT / $DURATION_WITH}")
    REDUCTION=$(awk "BEGIN {printf \"%.1f\", (1 - $DURATION_WITH / $DURATION_WITHOUT) * 100}")
    
    echo "训练时间对比:"
    echo "  不使用剪枝: ${DURATION_WITHOUT} 秒 ($(($DURATION_WITHOUT / 60)) 分钟)"
    echo "  使用剪枝:   ${DURATION_WITH} 秒 ($(($DURATION_WITH / 60)) 分钟)"
    echo ""
    echo "加速效果:"
    echo "  加速倍数:   ${SPEEDUP}x"
    echo "  时间减少:   ${REDUCTION}%"
fi

if [ -n "$POINTS_WITHOUT" ] && [ -n "$POINTS_WITH" ] && [ $POINTS_WITHOUT -gt 0 ]; then
    POINTS_REDUCTION=$(awk "BEGIN {printf \"%.1f\", (1 - $POINTS_WITH / $POINTS_WITHOUT) * 100}")
    
    echo ""
    echo "点云数量对比:"
    echo "  不使用剪枝: ${POINTS_WITHOUT} 个点"
    echo "  使用剪枝:   ${POINTS_WITH} 个点"
    echo "  减少比例:   ${POINTS_REDUCTION}%"
fi

echo ""
echo "=========================================="
echo "论文中的预期效果 (Instant4D):"
echo "  - 点云减少: 约 92%"
echo "  - 训练加速: 约 4倍"
echo "  - 渲染提升: 约 6倍"
echo "  - 内存减少: 约 90%"
echo "=========================================="

echo ""
echo "详细日志文件:"
echo "  - training_without_pruning.log (不使用剪枝)"
echo "  - training_with_pruning.log (使用剪枝)"

