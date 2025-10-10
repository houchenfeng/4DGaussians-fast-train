_base_ = './default.py'

# Debug配置 - 快速测试网格剪枝效果
# 使用少量迭代快速验证

ModelParams = dict(
    use_grid_pruning=True,  # 启用网格剪枝
)

OptimizationParams = dict(
    dataloader=True,
    iterations=500,           # 减少到500次迭代（原14000）
    batch_size=2,
    coarse_iterations=200,    # 粗糙阶段200次（原3000）
    densify_until_iter=300,   # 密集化到300次（原10000）
    densification_interval=50, # 密集化间隔50次（原100）
    opacity_reset_interval=60000,
    opacity_threshold_coarse=0.005,
    opacity_threshold_fine_init=0.005,
    opacity_threshold_fine_after=0.005,
)

