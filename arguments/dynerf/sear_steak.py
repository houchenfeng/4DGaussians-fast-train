_base_ = './default.py'
ModelParams = dict(
    use_grid_pruning=False,  # 不启用 Instant4D 网格剪枝，
    use_isotropic_gaussian=False,  # 不启用 Instant4D 各向同性高斯
    use_simplified_rgb=False,  # 不启用 Instant4D 简化RGB 
    sh_degree=3,  # 完整RGB ， 简化rgb 设置为0
)
OptimizationParams = dict(
    batch_size=2,
)