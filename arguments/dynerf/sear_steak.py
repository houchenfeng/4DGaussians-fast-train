_base_ = './default.py'
ModelParams = dict(
    use_grid_pruning=True,  # 启用 Instant4D 网格剪枝，减少92%点云，4倍训练加速
)
OptimizationParams = dict(
    batch_size=2,
)