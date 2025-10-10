#!/bin/bash

# 快速测试网格剪枝 - 单次运行
# 使用用户提供的命令测试网格剪枝效果

echo "=========================================="
echo "Instant4D 网格剪枝快速测试"
echo "=========================================="

# 确保配置文件启用了网格剪枝
CONFIG_FILE="arguments/dynerf/sear_steak.py"

if grep -q "use_grid_pruning=True" "$CONFIG_FILE"; then
    echo "✓ 网格剪枝已启用"
else
    echo "⚠ 配置文件中未启用网格剪枝"
    echo "请确保 $CONFIG_FILE 中包含:"
    echo "  ModelParams = dict("
    echo "      use_grid_pruning=True,"
    echo "  )"
fi

echo ""
echo "开始训练..."
echo "输出日志: training2.log"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 运行训练
CUDA_VISIBLE_DEVICES=0 python train.py \
    -s /home/lt/2024/data/N3D/multipleview/sear_steak \
    --port 6017 \
    --ip 127.0.0.4 \
    --expname "dynerf/sear_steak_pruning" \
    --configs arguments/dynerf/sear_steak.py \
    --quiet

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ 训练完成"
else
    echo "⚠ 训练退出 (退出码: $EXIT_CODE)"
fi

echo ""
echo "训练时间: $DURATION 秒 ($(($DURATION / 60)) 分钟 $(($DURATION % 60)) 秒)"
echo "=========================================="

# 提取关键信息
if [ -f "training2.log" ]; then
    echo ""
    echo "关键信息摘要:"
    echo "----------------------------------------"
    
    # 网格剪枝信息
    if grep -q "\[Grid Pruning\]" training2.log; then
        echo "网格剪枝信息:"
        grep "\[Grid Pruning\]" training2.log | tail -4 | sed 's/^/  /'
    fi
    
    # 初始点云数量
    POINTS=$(grep "Number of points at initialisation" training2.log | tail -1 | awk '{print $NF}')
    if [ -n "$POINTS" ]; then
        echo "初始点云数量: $POINTS"
    fi
    
    echo "----------------------------------------"
    echo ""
    echo "完整日志: training2.log"
fi

echo ""
echo "论文预期效果 (Instant4D):"
echo "  - 点云减少: ~92%"
echo "  - 训练加速: ~4x"
echo "  - 渲染提升: ~6x"
echo "=========================================="

