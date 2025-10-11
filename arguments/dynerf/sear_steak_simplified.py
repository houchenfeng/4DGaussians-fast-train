_base_ = './sear_steak.py'
# 仅启用简化RGB
ModelParams = dict(
    use_grid_pruning=False,
    use_isotropic_gaussian=False,
    use_simplified_rgb=True,
    sh_degree=0,
)

