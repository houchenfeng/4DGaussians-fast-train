_base_ = './sear_steak.py'
# 启用全部Instant4D改进
ModelParams = dict(
    use_grid_pruning=True,
    use_isotropic_gaussian=True,
    use_simplified_rgb=True,
    sh_degree=0,
)

