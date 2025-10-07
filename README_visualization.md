# 4DGS训练时间可视化分析工具

这个工具可以帮助你分析4DGS训练过程中的时间分布和性能瓶颈。

## 功能特性

### 📊 可视化图表
1. **每轮用时曲线** - 显示训练过程中每轮迭代的用时变化
2. **操作占比分析** - 饼图显示各操作的平均用时占比
3. **操作趋势图** - 显示主要操作的用时变化趋势
4. **热力图** - 直观显示所有操作的用时分布

### 📈 分析内容
- 每轮训练的总用时
- 各操作（数据加载、渲染、损失计算、优化器等）的用时分布
- 训练效率变化趋势
- 性能瓶颈识别

## 使用方法

### 1. 快速开始
```bash
# 使用默认参数运行完整分析
./run_visualization.sh

# 指定时间报告文件和输出目录
./run_visualization.sh output/your_experiment/timing_report.json output/your_experiment/visualization
```

### 2. 单独运行特定图表
```bash
# 只显示每轮用时曲线
python3 visualize_timing.py output/debug_test_1/timing_report.json --curve

# 只显示操作占比饼图
python3 visualize_timing.py output/debug_test_1/timing_report.json --breakdown

# 只显示操作趋势图
python3 visualize_timing.py output/debug_test_1/timing_report.json --trends

# 只显示热力图
python3 visualize_timing.py output/debug_test_1/timing_report.json --heatmap
```

### 3. 生成所有图表并保存
```bash
python3 visualize_timing.py output/debug_test_1/timing_report.json --output output/debug_test_1/visualization
```

## 输出文件

运行完整分析后会生成以下文件：
- `iteration_timing_curve.png` - 每轮用时曲线图
- `operation_breakdown.png` - 操作占比饼图
- `operation_trends.png` - 操作趋势图
- `operation_heatmap.png` - 操作用时热力图

## 依赖要求

### 必需依赖
- Python 3.6+
- matplotlib
- numpy

### 可选依赖
- seaborn (用于更好的热力图显示，如果没有安装会自动使用matplotlib)

### 安装依赖
```bash
pip install matplotlib numpy
# 可选
pip install seaborn
```

## 示例输出

### 控制台输出
```
📊 4DGS训练时间分析报告
============================================================
总训练时间: 31.46 秒 (0.5 分钟)
平均每轮用时: 0.157 秒
最快轮次: 0.112 秒
最慢轮次: 0.234 秒
总轮次数: 20

各操作平均用时:
  Coarse Data Loading: 0.045 秒
  Coarse Render: 0.263 秒
  Coarse Loss Computation: 0.405 秒
  Coarse Optimizer Step: 0.065 秒
  Fine Data Loading: 0.043 秒
  Fine Render: 0.288 秒
  Fine Loss Computation: 0.437 秒
  Fine Optimizer Step: 0.090 秒

最终损失: 0.035619
最终PSNR: 25.24
============================================================
```

## 分析建议

### 性能优化
1. **识别瓶颈** - 查看操作占比图，找出最耗时的操作
2. **趋势分析** - 观察操作趋势图，看是否有性能退化
3. **效率监控** - 通过用时曲线监控训练效率变化

### 常见问题
1. **渲染时间过长** - 可能需要优化渲染参数或减少分辨率
2. **损失计算耗时** - 可能需要优化损失函数或减少计算复杂度
3. **数据加载慢** - 可能需要优化数据加载或使用更快的存储

## 注意事项

1. 确保时间报告文件存在且格式正确
2. 如果没有安装seaborn，热力图会使用matplotlib绘制
3. 图表会显示在屏幕上，如果指定了输出目录也会保存到文件
4. 建议在训练完成后运行分析，以获得完整的统计数据
