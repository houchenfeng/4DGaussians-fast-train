#!/bin/bash

# Debug模式测试网格剪枝
# 使用少量迭代快速验证效果

echo "=========================================="
echo "网格剪枝 Debug 测试 (少量迭代)"
echo "=========================================="

DATA_PATH="/home/lt/2024/data/N3D/multipleview/sear_steak"
PORT=6017
IP="127.0.0.4"

# 创建debug输出目录
DEBUG_OUTPUT="./debug_results"
mkdir -p "$DEBUG_OUTPUT"

# 检查数据路径
if [ ! -d "$DATA_PATH" ]; then
    echo "错误: 数据路径不存在: $DATA_PATH"
    exit 1
fi

# 测试1: 不使用网格剪枝
echo ""
echo "=========================================="
echo "测试1: 不使用网格剪枝 (baseline)"
echo "迭代次数: 500"
echo "=========================================="

EXP_NAME_1="debug/sear_steak_no_pruning"
LOG_FILE_1="$DEBUG_OUTPUT/log_no_pruning.txt"

echo "实验名称: $EXP_NAME_1"
echo "日志文件: $LOG_FILE_1"
echo ""

START_TIME_1=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python train.py \
    -s "$DATA_PATH" \
    --port $PORT \
    --ip $IP \
    --expname "$EXP_NAME_1" \
    --configs arguments/dynerf/sear_steak_debug_no_pruning.py \
    2>&1 | tee "$LOG_FILE_1"

END_TIME_1=$(date +%s)
DURATION_1=$((END_TIME_1 - START_TIME_1))

echo ""
echo "✓ 测试1完成"
echo "耗时: $DURATION_1 秒 ($(($DURATION_1 / 60))分$(($DURATION_1 % 60))秒)"

# 提取关键信息
POINTS_1=$(grep "Number of points at initialisation" "$LOG_FILE_1" | tail -1 | awk '{print $NF}')
echo "初始点云数量: ${POINTS_1:-未知}"

# 测试2: 使用网格剪枝
echo ""
echo "=========================================="
echo "测试2: 使用网格剪枝 (Instant4D)"
echo "迭代次数: 500"
echo "=========================================="

EXP_NAME_2="debug/sear_steak_with_pruning"
LOG_FILE_2="$DEBUG_OUTPUT/log_with_pruning.txt"

echo "实验名称: $EXP_NAME_2"
echo "日志文件: $LOG_FILE_2"
echo ""

START_TIME_2=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python train.py \
    -s "$DATA_PATH" \
    --port $PORT \
    --ip $IP \
    --expname "$EXP_NAME_2" \
    --configs arguments/dynerf/sear_steak_debug.py \
    2>&1 | tee "$LOG_FILE_2"

END_TIME_2=$(date +%s)
DURATION_2=$((END_TIME_2 - START_TIME_2))

echo ""
echo "✓ 测试2完成"
echo "耗时: $DURATION_2 秒 ($(($DURATION_2 / 60))分$(($DURATION_2 % 60))秒)"

# 提取关键信息
POINTS_2=$(grep "Number of points at initialisation" "$LOG_FILE_2" | tail -1 | awk '{print $NF}')
PRUNING_INFO=$(grep "\[Grid Pruning\]" "$LOG_FILE_2")

echo "网格剪枝信息:"
echo "$PRUNING_INFO" | sed 's/^/  /'
echo "初始点云数量: ${POINTS_2:-未知}"

# 生成对比报告
REPORT_FILE="$DEBUG_OUTPUT/comparison_report.txt"

echo ""
echo "=========================================="
echo "生成对比报告..."
echo "=========================================="

cat > "$REPORT_FILE" << EOF
========================================
网格剪枝 Debug 测试对比报告
========================================
测试日期: $(date '+%Y-%m-%d %H:%M:%S')
迭代次数: 500 (debug模式)

----------------------------------------
测试1: 不使用网格剪枝
----------------------------------------
实验名称: $EXP_NAME_1
输出路径: output/$EXP_NAME_1
日志文件: $LOG_FILE_1
训练时间: $DURATION_1 秒 ($(($DURATION_1 / 60))分$(($DURATION_1 % 60))秒)
初始点数: ${POINTS_1:-未知}

----------------------------------------
测试2: 使用网格剪枝
----------------------------------------
实验名称: $EXP_NAME_2
输出路径: output/$EXP_NAME_2
日志文件: $LOG_FILE_2
训练时间: $DURATION_2 秒 ($(($DURATION_2 / 60))分$(($DURATION_2 % 60))秒)
初始点数: ${POINTS_2:-未知}

网格剪枝详情:
$PRUNING_INFO

----------------------------------------
性能对比
----------------------------------------
EOF

# 计算加速比和减少比例
if [ -n "$DURATION_1" ] && [ -n "$DURATION_2" ] && [ $DURATION_1 -gt 0 ]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $DURATION_1 / $DURATION_2}")
    TIME_REDUCTION=$(awk "BEGIN {printf \"%.1f\", (1 - $DURATION_2 / $DURATION_1) * 100}")
    
    echo "训练时间对比:" >> "$REPORT_FILE"
    echo "  不使用剪枝: $DURATION_1 秒" >> "$REPORT_FILE"
    echo "  使用剪枝:   $DURATION_2 秒" >> "$REPORT_FILE"
    echo "  加速倍数:   ${SPEEDUP}x" >> "$REPORT_FILE"
    echo "  时间减少:   ${TIME_REDUCTION}%" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

if [ -n "$POINTS_1" ] && [ -n "$POINTS_2" ] && [ $POINTS_1 -gt 0 ]; then
    POINT_REDUCTION=$(awk "BEGIN {printf \"%.1f\", (1 - $POINTS_2 / $POINTS_1) * 100}")
    
    echo "点云数量对比:" >> "$REPORT_FILE"
    echo "  不使用剪枝: $POINTS_1 个点" >> "$REPORT_FILE"
    echo "  使用剪枝:   $POINTS_2 个点" >> "$REPORT_FILE"
    echo "  减少比例:   ${POINT_REDUCTION}%" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF
----------------------------------------
输出文件位置
----------------------------------------
对比报告:   $REPORT_FILE
测试1日志:  $LOG_FILE_1
测试2日志:  $LOG_FILE_2
测试1输出:  output/$EXP_NAME_1/
测试2输出:  output/$EXP_NAME_2/

渲染结果:
  测试1视频: output/$EXP_NAME_1/video/
  测试2视频: output/$EXP_NAME_2/video/
  测试1图像: output/$EXP_NAME_1/test/ours_500/
  测试2图像: output/$EXP_NAME_2/test/ours_500/

模型检查点:
  测试1: output/$EXP_NAME_1/point_cloud/iteration_500/
  测试2: output/$EXP_NAME_2/point_cloud/iteration_500/

----------------------------------------
论文预期效果 (Instant4D)
----------------------------------------
点云减少: ~92%
训练加速: ~4x
渲染提升: ~6x
内存减少: ~90%

========================================
EOF

# 显示报告
echo ""
cat "$REPORT_FILE"
echo ""

echo "=========================================="
echo "✓ Debug测试完成！"
echo "=========================================="
echo ""
echo "📊 详细报告: $REPORT_FILE"
echo ""
echo "📁 输出目录:"
echo "  - debug_results/          (日志和报告)"
echo "  - output/$EXP_NAME_1/     (不使用剪枝)"
echo "  - output/$EXP_NAME_2/     (使用剪枝)"
echo ""
echo "查看日志:"
echo "  cat $LOG_FILE_1"
echo "  cat $LOG_FILE_2"
echo ""

