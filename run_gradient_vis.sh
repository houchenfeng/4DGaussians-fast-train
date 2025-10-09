#!/bin/bash

# 从已保存的checkpoint生成梯度时间轴可视化的便捷脚本
# Usage: ./run_gradient_vis.sh [model_path] [iteration]

# 默认参数
DEFAULT_MODEL_PATH="/home/lt/2024/fast4dgs/4DGaussians-fast-train/output/dynerf/sear_steak_test"
DEFAULT_ITERATION=14000

# 获取参数
MODEL_PATH=${1:-$DEFAULT_MODEL_PATH}
ITERATION=${2:-$DEFAULT_ITERATION}

echo "============================================"
echo "Gradient Timeline Visualization from Checkpoint"
echo "============================================"
echo "Model path: ${MODEL_PATH}"
echo "Iteration: ${ITERATION}"
echo ""

# 检查模型路径是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model path does not exist: ${MODEL_PATH}"
    exit 1
fi

# 检查checkpoint是否存在
CHECKPOINT_PATH="${MODEL_PATH}/point_cloud/iteration_${ITERATION}/point_cloud.ply"
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint not found: ${CHECKPOINT_PATH}"
    echo ""
    echo "Available iterations:"
    ls -d ${MODEL_PATH}/point_cloud/iteration_* 2>/dev/null | xargs -n 1 basename
    exit 1
fi

echo "✓ Checkpoint found: ${CHECKPOINT_PATH}"
echo ""

# 设置输出目录
OUTPUT_DIR="${MODEL_PATH}/gradvis/iteration_${ITERATION}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# 运行可视化脚本
echo "Generating gradient timeline visualization..."
echo "This will compute gradients at 10 time points (t=0, 0.1, ..., 0.9)"
echo ""

python visualize_gradient_from_checkpoint.py \
    --model_path "${MODEL_PATH}" \
    --iteration ${ITERATION} \
    --output_dir "${OUTPUT_DIR}" \
    --max_points 10000

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ Visualization Complete!"
    echo "============================================"
    echo ""
    echo "Generated files:"
    ls -lh ${OUTPUT_DIR}/gradient_3d/*.html 2>/dev/null
    echo ""
    echo "Open in browser:"
    echo "  firefox ${OUTPUT_DIR}/gradient_3d/gradient_timeline.html"
    echo ""
    echo "Or use absolute path:"
    ABSOLUTE_PATH=$(readlink -f ${OUTPUT_DIR}/gradient_3d/gradient_timeline.html)
    echo "  firefox ${ABSOLUTE_PATH}"
    echo ""
    echo "Interactive Controls (in HTML):"
    echo "  • Point Size: Adjust point cloud (0.1x - 5x)"
    echo "  • Arrow Size: Adjust arrow display size (0.5x - 16x)"
    echo "  • Visibility: Show Both / Points Only / Arrows Only"
    echo "  • Time Slider: View gradients at different time points"
    echo "  • Play/Pause: Automatic animation"
    echo ""
    echo "Visualization:"
    echo "  • Arrow direction = Gradient direction (normalized)"
    echo "  • Color (point & arrow) = Gradient magnitude"
    echo "  • Arrow length = Uniform (adjustable via dropdown)"
    echo ""
    echo "Notes:"
    echo "  - All settings persist during Play/Slider changes (JS-based)"
    echo "  - Open browser console (F12) to see debug logs"
else
    echo ""
    echo "Error: Visualization failed"
    exit 1
fi

