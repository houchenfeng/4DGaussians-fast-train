_base_ = './default.py'

# 极速调试配置 - 仅用于验证代码修改，不关注训练质量
ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 8,   # 大幅减少输出维度
     'resolution': [16, 16, 16, 20]  # 极低分辨率
    },
    multires = [1],  # 减少多分辨率
    defor_depth = 0,
    net_width = 32,   # 极小网络宽度
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,  # 关闭渲染过程
    static_mlp=False
)

OptimizationParams = dict(
    dataloader=True,
    iterations = 100,         # 极少的迭代次数
    batch_size=1,           # 最小批次
    coarse_iterations = 100,   # 极少的粗调迭代
    densify_until_iter = 8,
    opacity_reset_interval = 1000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # 关闭一些耗时的操作
    pruning_interval = 1000,
    densification_interval = 5,
)
