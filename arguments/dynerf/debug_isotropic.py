_base_ = './debug_test.py'
# Debug测试 - 各向同性高斯
ModelParams = dict(
    use_grid_pruning=False,
    use_isotropic_gaussian=True,
    use_simplified_rgb=False,
)

