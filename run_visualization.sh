#!/bin/bash

# 4DGS训练时间可视化分析脚本
# 使用方法: ./run_visualization.sh [timing_report.json] [output_dir]

TIMING_REPORT=${1:-"output/debug_test_1/timing_report.json"}
OUTPUT_DIR=${2:-"output/debug_test_1/visualization"}

echo "=== 4DGS训练时间可视化分析 ==="
echo "时间报告文件: $TIMING_REPORT"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查文件是否存在
if [ ! -f "$TIMING_REPORT" ]; then
    echo "❌ 错误: 时间报告文件不存在: $TIMING_REPORT"
    echo "请先运行训练生成时间报告，或指定正确的文件路径"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行可视化分析
echo "🚀 开始可视化分析..."
python3 visualize_timing.py "$TIMING_REPORT" --output "$OUTPUT_DIR"

echo ""
echo "✅ 可视化分析完成！"
echo "📁 结果保存在: $OUTPUT_DIR"
echo ""
echo "生成的文件:"
echo "  - iteration_timing_curve.png  (每轮用时曲线)"
echo "  - operation_breakdown.png     (操作占比饼图)"
echo "  - operation_trends.png        (操作趋势图)"
echo "  - operation_heatmap.png       (操作用时热力图)"
echo ""
echo "💡 提示: 你也可以单独运行特定图表:"
echo "  python3 visualize_timing.py $TIMING_REPORT --curve     # 只显示用时曲线"
echo "  python3 visualize_timing.py $TIMING_REPORT --breakdown # 只显示操作占比"
echo "  python3 visualize_timing.py $TIMING_REPORT --trends    # 只显示操作趋势"
echo "  python3 visualize_timing.py $TIMING_REPORT --heatmap   # 只显示热力图"
