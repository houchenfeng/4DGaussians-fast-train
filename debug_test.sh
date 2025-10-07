#!/bin/bash

# 极速调试测试脚本 - 仅用于验证代码修改
# 10次迭代，预期训练时间: 30秒-1分钟

echo "=== 极速调试测试模式 ==="
echo "训练配置: 100次迭代 (100粗调 + 100精细调)"
echo "预期训练时间: 30秒-1分钟"
echo "用途: 验证代码修改，测试计时功能"
echo "=========================="

# 手动指定数字，端口、IP和实验名都基于这个数字
# 使用方法: ./debug_test.sh [NUMBER]
# 例如: ./debug_test.sh 1  (端口6019+1=6020, IP 127.0.0.1, 实验名debug_test_1)
# 例如: ./debug_test.sh 5  (端口6019+5=6024, IP 127.0.0.5, 实验名debug_test_5)

NUMBER=${1:-1}        # 默认数字1，可通过第一个参数指定
TEST_PORT=$((6019 + NUMBER))  # 端口 = 6019 + 数字
TEST_IP="127.0.0.${NUMBER}"   # IP = 127.0.0.数字
TEST_EXPNAME="debug_test_${NUMBER}"  # 实验名 = debug_test_数字

# 设置测试参数
TEST_DATASET="/home/lt/2024/data/N3D/multipleview/sear_steak"

echo "使用数字: ${NUMBER}"
echo "使用端口: ${TEST_PORT} (6019 + ${NUMBER})"
echo "使用IP: ${TEST_IP}"
echo "实验名: ${TEST_EXPNAME}"

TEST_CONFIG="arguments/dynerf/debug_test.py"

# 清理之前的测试结果
echo "清理之前的测试结果..."
rm -rf "output/${TEST_EXPNAME}"

# 开始极速测试训练
echo "开始极速调试测试..."
echo "数据集: ${TEST_DATASET}"
echo "配置: ${TEST_CONFIG}"
echo "输出目录: output/${TEST_EXPNAME}"
echo ""

# 记录开始时间
start_time=$(date +%s)

CUDA_VISIBLE_DEVICES=0 python train.py \
    -s "${TEST_DATASET}" \
    --port ${TEST_PORT} \
    --ip ${TEST_IP} \
    --expname "${TEST_EXPNAME}" \
    --configs "${TEST_CONFIG}" \
    --debug_mode \
    

# 计算实际耗时
end_time=$(date +%s)
actual_time=$((end_time - start_time))

echo ""
echo "=== 极速调试测试完成 ==="
echo "实际耗时: ${actual_time} 秒"
echo "检查输出目录: output/${TEST_EXPNAME}"
echo ""

# 显示计时报告摘要
if [ -f "output/${TEST_EXPNAME}/timing_report.json" ]; then
    echo "=== 计时摘要 ==="
    python3 -c "
import json
import sys
import os

try:
    file_path = 'output/${TEST_EXPNAME}/timing_report.json'
    print(f'正在读取文件: {file_path}')
    
    if not os.path.exists(file_path):
        print(f'错误: 文件不存在 {file_path}')
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'总训练时间: {data.get(\"total_training_time\", 0):.1f} 秒')
    
    if 'training_logs' in data and data['training_logs']:
        logs = data['training_logs']
        print(f'训练日志条目: {len(logs)}')
        if logs:
            final_log = logs[-1]
            print(f'最终损失: {final_log.get(\"loss\", \"N/A\"):.6f}')
            if 'psnr' in final_log and final_log['psnr'] is not None:
                print(f'最终PSNR: {final_log[\"psnr\"]:.2f}')
    
    print('\\n各阶段耗时:')
    detailed_timings = data.get('detailed_timings', {})
    for category, timings in detailed_timings.items():
        if isinstance(timings, dict):
            print(f'  {category}:')
            for name, timing in timings.items():
                if isinstance(timing, dict) and 'total_elapsed' in timing:
                    print(f'    {name}: {timing[\"total_elapsed\"]:.2f}s ({timing.get(\"percentage\", 0):.1f}%)')
        else:
            # 兼容旧的格式
            if 'total_elapsed' in timings:
                print(f'  {category}: {timings[\"total_elapsed\"]:.2f}s ({timings.get(\"percentage\", 0):.1f}%)')
    
    # 显示每轮用时统计
    per_iteration = data.get('per_iteration_timings', {})
    if per_iteration:
        print('\\n每轮用时统计:')
        iteration_times = []
        for iter_num, timings in per_iteration.items():
            if isinstance(timings, dict):
                for key, time_val in timings.items():
                    if key.endswith('_total'):
                        iteration_times.append((int(iter_num), time_val))
                        break
        
        if iteration_times:
            iteration_times.sort(key=lambda x: x[0])
            avg_time = sum(time for _, time in iteration_times) / len(iteration_times)
            min_time = min(time for _, time in iteration_times)
            max_time = max(time for _, time in iteration_times)
            print(f'  平均每轮用时: {avg_time:.3f}s')
            print(f'  最快轮次: {min_time:.3f}s, 最慢轮次: {max_time:.3f}s')
            print(f'  总轮次数: {len(iteration_times)}')
                
except json.JSONDecodeError as e:
    print(f'JSON解析错误: {e}')
    print('文件内容可能损坏或格式不正确')
except FileNotFoundError as e:
    print(f'文件未找到: {e}')
except Exception as e:
    print(f'读取计时报告失败: {e}')
    import traceback
    traceback.print_exc()
"
else
    echo "未找到计时报告文件"
fi

echo ""
echo "调试测试完成！"
echo "如需查看详细结果:"
echo "cat output/${TEST_EXPNAME}/timing_report.json"
