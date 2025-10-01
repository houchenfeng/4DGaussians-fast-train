# 新增功能说明

本文档说明了为4DGaussians训练脚本新增的计时功能和debug功能。

## 1. 计时功能

### 功能描述
- 自动计算训练过程中各个部分的用时
- 包括：渲染时间、损失计算时间、densification时间、优化器步骤时间等
- 生成详细的计时报告并保存到输出目录

### 使用方法
计时功能会自动启用，无需额外参数。训练结束后会在输出目录生成 `timing_report.json` 文件。

### 输出文件
- `timing_report.json`: 包含详细的计时统计信息
  - 总训练时间
  - 各操作的总用时、调用次数、平均用时

### 示例输出
```json
{
  "total_training_time": 3600.5,
  "detailed_timings": {
    "coarse_render": {
      "total_elapsed": 1200.3,
      "call_count": 1000,
      "average_per_call": 1.2003
    },
    "coarse_loss_computation": {
      "total_elapsed": 800.2,
      "call_count": 1000,
      "average_per_call": 0.8002
    }
  }
}
```

## 2. Debug功能

### 功能描述
- 在debug模式下保存训练图像和真实图像的拼接图
- 帮助可视化训练过程中的渲染质量变化
- 自动计算帧数信息

### 使用方法
添加 `--debug_mode` 参数启用debug功能：

```bash
python train.py --debug_mode [其他参数...]
```

### 输出文件
- `debug/` 目录：包含所有debug图像
- 图像命名格式：`{stage}_{iter}_{cam.image_name}_{frame_number}.png`
  - `stage`: 训练阶段（coarse/fine）
  - `iter`: 迭代次数
  - `cam.image_name`: 相机/图像名称
  - `frame_number`: 计算得出的帧数（int(cam.time * maxtime)）

### 示例文件名
- `coarse_100_image_001_50.png`: coarse阶段，第100次迭代，图像001，第50帧
- `fine_5000_image_045_120.png`: fine阶段，第5000次迭代，图像045，第120帧

### 图像格式
- 水平拼接：[渲染图像] | [真实图像]
- 白色分隔线分隔两部分
- PNG格式，便于查看和比较

## 3. 技术实现细节

### 计时器类
- `DetailedTimer`: 新增的详细计时器类
- 支持多个并发计时器
- 自动统计调用次数和平均时间

### Debug工具
- `save_debug_image()`: 保存拼接图像
- `create_debug_summary()`: 创建debug摘要
- 自动处理不同数据集格式（PanopticSports等）

### 帧数计算
- 使用公式：`int(cam.time * maxtime)`
- `cam.time`: 0-1之间的标准化时间值
- `maxtime`: 场景的最大时间长度

## 4. 使用示例

### 基本训练（带计时）
```bash
python train.py --source_path /path/to/data --model_path /path/to/output
```

### 启用debug模式
```bash
python train.py --source_path /path/to/data --model_path /path/to/output --debug_mode
```

### 查看结果
训练完成后，检查输出目录：
- `timing_report.json`: 计时报告
- `debug/`: debug图像目录
- `debug/*_debug_summary.txt`: debug摘要

## 5. 注意事项

1. **性能影响**: debug模式会增加磁盘I/O，建议仅在需要时启用
2. **存储空间**: debug图像会占用额外存储空间
3. **频率控制**: 当前设置为每100次迭代保存一次debug图像
4. **兼容性**: 支持所有现有的数据集格式

## 6. 故障排除

### 常见问题
1. **debug图像保存失败**: 检查磁盘空间和写入权限
2. **计时报告为空**: 确保训练正常完成
3. **帧数计算错误**: 检查maxtime值是否正确

### 调试建议
- 查看控制台输出的错误信息
- 检查输出目录的权限设置
- 验证数据集的时间信息是否正确
