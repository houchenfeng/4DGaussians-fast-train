_base_ = './debug_test.py'
# Debug测试 - 网格剪枝
ModelParams = dict(
    use_grid_pruning=True,
    use_isotropic_gaussian=False,
    use_simplified_rgb=False,
)

