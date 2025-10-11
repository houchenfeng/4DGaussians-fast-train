_base_ = './sear_steak.py'
# 仅启用各向同性高斯
ModelParams = dict(
    use_grid_pruning=False,
    use_isotropic_gaussian=True,
    use_simplified_rgb=False,
)

