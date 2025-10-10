_base_ = './default.py'

# Debug配置 - 不使用网格剪枝的对照组
# 使用少量迭代快速验证

ModelParams = dict(
    use_grid_pruning=False,  # 禁用网格剪枝
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

