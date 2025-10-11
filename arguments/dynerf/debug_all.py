_base_ = './debug_test.py'
# Debug测试 - 全部改进
ModelParams = dict(
    use_grid_pruning=True,
    use_isotropic_gaussian=True,
    use_simplified_rgb=True,
    sh_degree=0,
)

