_base_ = './debug_test.py'
# Debug测试 - 简化RGB
ModelParams = dict(
    use_grid_pruning=False,
    use_isotropic_gaussian=False,
    use_simplified_rgb=True,
    sh_degree=0,
)

