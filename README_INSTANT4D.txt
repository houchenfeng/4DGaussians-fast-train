================================================================================
                    Instant4D改进模块 - 使用说明
================================================================================

核心模块 (3个):
  utils/grid_pruning.py          网格剪枝 (点云↓70-95%, 训练↑4x)
  utils/isotropic_gaussian.py    各向同性高斯 (稳定性↑, PSNR+1.25dB)
  utils/simplified_rgb.py        简化RGB (参数↓93.75%)

测试脚本 (3个):
  verify_improvements.py         快速验证 (1分钟)
  debug_test.sh                  Debug测试 (5-10分钟, 继承debug_test.py)
  run.sh [模块名]                分模块完整测试 (40-90分钟/模块)

================================================================================
                        🚀 快速使用
================================================================================

1. 验证模块 (1分钟)
   python verify_improvements.py

2. Debug测试所有模块 (5-10分钟)
   ./debug_test.sh
   
   配置继承: arguments/dynerf/debug_test.py (300次迭代)
   GPU: 1
   
   测试5个配置:
   1. baseline    - 无改进
   2. pruning     - 仅网格剪枝
   3. isotropic   - 仅各向同性
   4. simplified  - 仅简化RGB
   5. all         - 全部改进
   
   输出: output/debug/{baseline,pruning,isotropic,simplified,all}/

3. 完整测试 - 分模块运行 (40-90分钟/模块)
   ./run.sh baseline       - 测试Baseline
   ./run.sh pruning        - 测试网格剪枝 (推荐)
   ./run.sh isotropic      - 测试各向同性
   ./run.sh simplified     - 测试简化RGB
   ./run.sh all            - 测试全部改进
   
   配置基础: arguments/dynerf/sear_steak.py (14000次迭代)
   GPU: 1
   
   每个训练后自动:
   - render.py: 渲染测试图像
   - metrics.py: 计算PSNR/SSIM
   
   输出: output/full/{模块名}/
   指标: output/full_{模块名}_metrics.txt

================================================================================
                        ⚙️ 配置文件
================================================================================

Debug测试配置 (继承debug_test.py):
  arguments/dynerf/debug_test.py        - 基础配置 (300iter)
  arguments/dynerf/debug_baseline.py    - Baseline (自动生成)
  arguments/dynerf/debug_pruning.py     - 网格剪枝 (自动生成)
  arguments/dynerf/debug_isotropic.py   - 各向同性 (自动生成)
  arguments/dynerf/debug_simplified.py  - 简化RGB (自动生成)
  arguments/dynerf/debug_all.py         - 全部改进 (自动生成)

完整测试配置 (继承sear_steak.py):
  arguments/dynerf/sear_steak.py        - Baseline配置 (14000iter)
  arguments/dynerf/sear_steak_pruning.py    - 仅网格剪枝
  arguments/dynerf/sear_steak_all.py        - 全部改进
  arguments/dynerf/run_isotropic.py     - 仅各向同性 (自动生成)
  arguments/dynerf/run_simplified.py    - 仅简化RGB (自动生成)

================================================================================
                        📊 使用示例
================================================================================

示例1: Debug快速测试
---------------------
./debug_test.sh

自动测试所有5个配置，生成对比表格

示例2: 测试单个改进
-------------------
# 仅测试网格剪枝 (推荐首先测试)
./run.sh pruning

# 仅测试各向同性
./run.sh isotropic

# 仅测试简化RGB
./run.sh simplified

示例3: 测试全部改进
-------------------
./run.sh all

示例4: 对比Baseline和改进
-------------------------
# 先运行Baseline
./run.sh baseline

# 然后运行网格剪枝
./run.sh pruning

# 对比结果
cat output/full_baseline_metrics.txt
cat output/full_pruning_metrics.txt

示例5: 在自己的配置中启用
------------------------
# 编辑你的配置文件
vim arguments/dynerf/your_config.py

# 添加改进参数
ModelParams = dict(
    use_grid_pruning=True,         # 网格剪枝
    use_isotropic_gaussian=False,
    use_simplified_rgb=False,
    sh_degree=3,
)

# 运行训练
python train.py -s /path/to/data --configs arguments/dynerf/your_config.py

================================================================================
                        📁 输出目录结构
================================================================================

Debug测试 (300iter):
  output/debug/baseline/
  output/debug/pruning/
  output/debug/isotropic/
  output/debug/simplified/
  output/debug/all/

完整测试 (14000iter):
  output/full/baseline/
    ├── point_cloud/iteration_14000/      - 模型检查点
    ├── test/ours_14000/
    │   ├── renders/                      - 渲染图像
    │   └── gt/                           - Ground truth
    └── cfg_args                          - 配置

  output/full/pruning/                    - 网格剪枝结果
  output/full/isotropic/                  - 各向同性结果
  output/full/simplified/                 - 简化RGB结果
  output/full/all/                        - 全部改进结果

训练日志:
  output/full_baseline.log
  output/full_pruning.log
  output/full_isotropic.log
  output/full_simplified.log
  output/full_all.log

指标文件:
  output/full_baseline_metrics.txt        - PSNR, SSIM, LPIPS等
  output/full_pruning_metrics.txt
  output/full_isotropic_metrics.txt
  output/full_simplified_metrics.txt
  output/full_all_metrics.txt

================================================================================
                        📈 查看结果
================================================================================

# 查看训练时间
grep "训练完成" output/full_*.log

# 查看PSNR指标
grep "PSNR" output/full_*_metrics.txt

# 查看网格剪枝效果
grep "\[Grid Pruning\]" output/full_pruning.log

# 查看渲染图像
ls -lh output/full/pruning/test/ours_14000/renders/

# 对比不同配置的渲染质量
# 可以用图像查看器对比：
# output/full/baseline/test/ours_14000/renders/00000.png
# output/full/pruning/test/ours_14000/renders/00000.png

================================================================================
                        ⚠️ 重要配置说明
================================================================================

关键参数: densify_until_iter
-----------------------------
必须接近 iterations，否则会导致性能下降！

✓ 正确配置:
  iterations=14000
  densify_until_iter=10000    # 接近iterations

✗ 错误配置:
  iterations=14000
  densify_until_iter=60       # 太小! 会导致点云无法增长

原因:
  - Instant4D初始点少 (网格剪枝后约5000点)
  - 需要通过密集化增长到合理数量 (1-2万点)
  - 过早停止密集化会导致点云被锁定
  - 点太少 → 表达能力不足 → 训练更慢

推荐配置:
----------
仅网格剪枝 (最保守，加速明显):
  use_grid_pruning=True
  sh_degree=3
  densify_until_iter=10000

全部改进 (需要调优):
  use_grid_pruning=True
  use_isotropic_gaussian=True
  use_simplified_rgb=True
  sh_degree=0
  densify_until_iter=10000

================================================================================
                        💡 使用建议
================================================================================

1. 首次使用:
   - 运行 verify_improvements.py 验证环境
   - 运行 ./debug_test.sh 快速测试
   - 运行 ./run.sh pruning 测试网格剪枝

2. 性能评估:
   - 先测试 baseline: ./run.sh baseline
   - 再测试 pruning: ./run.sh pruning
   - 对比训练时间和PSNR

3. 实际应用:
   - 推荐仅启用网格剪枝
   - 在自己的配置中设置 use_grid_pruning=True
   - 确保 densify_until_iter 设置正确

================================================================================

快速开始:
  python verify_improvements.py    验证模块
  ./debug_test.sh                  Debug测试所有
  ./run.sh pruning                 完整测试单个模块

查看帮助:
  ./run.sh                         显示用法说明

================================================================================
