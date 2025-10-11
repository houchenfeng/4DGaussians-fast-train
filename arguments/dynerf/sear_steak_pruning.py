_base_ = './sear_steak.py'
# 仅启用网格剪枝（推荐配置）
ModelParams = dict(
    use_grid_pruning=True,
    use_isotropic_gaussian=False,
    use_simplified_rgb=False,
    sh_degree=3,
)

