"""
网格剪枝模块 - Grid Pruning
基于 Instant4D 论文实现，用于减少点云数量，提升训练速度

主要功能:
1. 体素下采样 - 使用自适应体素大小进行点云过滤
2. 减少92%的高斯数量，4倍训练加速，6倍渲染提升
"""

import numpy as np
import open3d as o3d
from utils.graphics_utils import BasicPointCloud
import torch


def voxel_downsample(points, colors, voxel_size):
    """
    使用体素下采样减少点云数量
    
    Args:
        points: np.array (N, 3) - 点云坐标
        colors: np.array (N, 3) - 点云颜色
        voxel_size: float - 体素大小
        
    Returns:
        downsampled_points: np.array - 下采样后的点云
        downsampled_colors: np.array - 下采样后的颜色
    """
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 体素下采样
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # 转换回 numpy 数组
    downsampled_points = np.asarray(pcd_down.points)
    downsampled_colors = np.asarray(pcd_down.colors)
    
    return downsampled_points, downsampled_colors


def compute_adaptive_voxel_size(points, cameras=None, focal_mean=None, depth_mean=None, scale_factor=3.0):
    """
    计算自适应体素大小
    
    根据 Instant4D 论文:
    voxel_size = mean_depth / focal * scale_factor
    
    Args:
        points: np.array (N, 3) - 点云坐标
        cameras: list - 相机列表 (可选)
        focal_mean: float - 平均焦距 (可选)
        depth_mean: float - 平均深度 (可选)
        scale_factor: float - 缩放因子，论文中动态区域用3，静态区域用4
        
    Returns:
        voxel_size: float - 计算得到的体素大小
    """
    # 如果提供了相机信息，计算平均深度和焦距
    if cameras is not None and len(cameras) > 0:
        depths = []
        focals = []
        
        for cam in cameras:
            # 计算相机中心
            cam_center = -np.matmul(cam.R.T, cam.T)
            # 计算点到相机的平均深度
            dists = np.linalg.norm(points - cam_center.reshape(1, 3), axis=1)
            depths.append(np.median(dists))
            
            # 获取焦距
            # FovX = 2 * atan(width / (2 * fx))
            # fx = width / (2 * tan(FovX/2))
            fx = cam.width / (2 * np.tan(cam.FovX / 2))
            fy = cam.height / (2 * np.tan(cam.FovY / 2))
            focals.append((fx + fy) / 2)
        
        depth_mean = np.mean(depths) if depth_mean is None else depth_mean
        focal_mean = np.mean(focals) if focal_mean is None else focal_mean
    
    # 如果没有提供深度和焦距，使用默认值
    if depth_mean is None:
        depth_mean = np.percentile(np.linalg.norm(points - points.mean(axis=0), axis=1), 50)
    if focal_mean is None:
        focal_mean = 1000.0  # 默认焦距
    
    # 计算体素大小: voxel_size = mean_depth / focal * scale_factor
    voxel_size = depth_mean / focal_mean * scale_factor
    
    # 确保体素大小在合理范围内
    voxel_size = max(voxel_size, 0.001)  # 最小值
    voxel_size = min(voxel_size, 1.0)     # 最大值
    
    return voxel_size


def grid_pruning(pcd, cameras=None, use_adaptive=True, static_scale=4.0, dynamic_scale=3.0):
    """
    网格剪枝 - 主函数
    
    基于 Instant4D 论文的 Grid Pruning 策略
    - 使用自适应体素大小进行下采样
    - 减少92%的点云数量
    - 4倍训练加速，6倍渲染提升
    
    Args:
        pcd: BasicPointCloud - 输入点云
        cameras: list - 相机列表 (可选，用于计算自适应体素大小)
        use_adaptive: bool - 是否使用自适应体素大小
        static_scale: float - 静态区域缩放因子 (论文中为4)
        dynamic_scale: float - 动态区域缩放因子 (论文中为3)
        
    Returns:
        pruned_pcd: BasicPointCloud - 剪枝后的点云
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    
    print(f"[Grid Pruning] 原始点云数量: {points.shape[0]}")
    
    if use_adaptive and cameras is not None:
        # 使用自适应体素大小
        # 论文中使用 mean_depth / focal * scale_factor
        # 这里我们使用一个折中的 scale_factor
        avg_scale = (static_scale + dynamic_scale) / 2
        voxel_size = compute_adaptive_voxel_size(points, cameras, scale_factor=avg_scale)
        print(f"[Grid Pruning] 自适应体素大小: {voxel_size:.6f}")
    else:
        # 使用固定体素大小
        # 根据点云的尺度自动计算
        bbox_diagonal = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        voxel_size = bbox_diagonal / 100.0  # 将空间分成约100^3的网格
        print(f"[Grid Pruning] 固定体素大小: {voxel_size:.6f}")
    
    # 体素下采样
    downsampled_points, downsampled_colors = voxel_downsample(points, colors, voxel_size)
    
    # 法向量处理：简单地重新计算或置零
    if normals.shape[0] > 0:
        # 使用最近邻插值法向量
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        _, indices = tree.query(downsampled_points, k=1)
        downsampled_normals = normals[indices]
    else:
        downsampled_normals = np.zeros_like(downsampled_points)
    
    print(f"[Grid Pruning] 剪枝后点云数量: {downsampled_points.shape[0]}")
    print(f"[Grid Pruning] 点云减少比例: {(1 - downsampled_points.shape[0] / points.shape[0]) * 100:.1f}%")
    
    # 创建新的点云对象
    pruned_pcd = BasicPointCloud(
        points=downsampled_points,
        colors=downsampled_colors,
        normals=downsampled_normals
    )
    
    return pruned_pcd


def grid_pruning_with_motion(pcd, motion_prob=None, cameras=None, 
                              static_scale=4.0, dynamic_scale=3.0, threshold=0.7):
    """
    带运动感知的网格剪枝
    
    论文中的完整版本，区分静态和动态区域，使用不同的体素大小
    
    Args:
        pcd: BasicPointCloud - 输入点云
        motion_prob: np.array - 运动概率 (N,) (可选)
        cameras: list - 相机列表 (可选)
        static_scale: float - 静态区域缩放因子 (论文中为4)
        dynamic_scale: float - 动态区域缩放因子 (论文中为3)
        threshold: float - 动态/静态分割阈值 (论文中为0.7)
        
    Returns:
        pruned_pcd: BasicPointCloud - 剪枝后的点云
        pruned_motion_prob: np.array - 剪枝后的运动概率 (如果提供了motion_prob)
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals)
    
    print(f"[Grid Pruning with Motion] 原始点云数量: {points.shape[0]}")
    
    if motion_prob is not None:
        # 分离动态和静态区域
        dynamic_mask = motion_prob > threshold
        static_mask = ~dynamic_mask
        
        print(f"[Grid Pruning with Motion] 动态点: {dynamic_mask.sum()}, 静态点: {static_mask.sum()}")
        
        # 分别处理动态和静态区域
        all_downsampled_points = []
        all_downsampled_colors = []
        all_downsampled_normals = []
        all_downsampled_motion = []
        
        # 处理静态区域
        if static_mask.sum() > 0:
            static_points = points[static_mask]
            static_colors = colors[static_mask]
            voxel_size_static = compute_adaptive_voxel_size(
                static_points, cameras, scale_factor=static_scale
            )
            print(f"[Grid Pruning with Motion] 静态区域体素大小: {voxel_size_static:.6f}")
            
            down_points, down_colors = voxel_downsample(
                static_points, static_colors, voxel_size_static
            )
            all_downsampled_points.append(down_points)
            all_downsampled_colors.append(down_colors)
            all_downsampled_normals.append(np.zeros((down_points.shape[0], 3)))
            all_downsampled_motion.append(np.zeros(down_points.shape[0]))
        
        # 处理动态区域
        if dynamic_mask.sum() > 0:
            dynamic_points = points[dynamic_mask]
            dynamic_colors = colors[dynamic_mask]
            voxel_size_dynamic = compute_adaptive_voxel_size(
                dynamic_points, cameras, scale_factor=dynamic_scale
            )
            print(f"[Grid Pruning with Motion] 动态区域体素大小: {voxel_size_dynamic:.6f}")
            
            down_points, down_colors = voxel_downsample(
                dynamic_points, dynamic_colors, voxel_size_dynamic
            )
            all_downsampled_points.append(down_points)
            all_downsampled_colors.append(down_colors)
            all_downsampled_normals.append(np.zeros((down_points.shape[0], 3)))
            all_downsampled_motion.append(np.ones(down_points.shape[0]))
        
        # 合并
        downsampled_points = np.vstack(all_downsampled_points)
        downsampled_colors = np.vstack(all_downsampled_colors)
        downsampled_normals = np.vstack(all_downsampled_normals)
        downsampled_motion = np.concatenate(all_downsampled_motion)
        
    else:
        # 没有运动信息，统一处理
        avg_scale = (static_scale + dynamic_scale) / 2
        voxel_size = compute_adaptive_voxel_size(points, cameras, scale_factor=avg_scale)
        print(f"[Grid Pruning with Motion] 统一体素大小: {voxel_size:.6f}")
        
        downsampled_points, downsampled_colors = voxel_downsample(points, colors, voxel_size)
        downsampled_normals = np.zeros_like(downsampled_points)
        downsampled_motion = None
    
    print(f"[Grid Pruning with Motion] 剪枝后点云数量: {downsampled_points.shape[0]}")
    print(f"[Grid Pruning with Motion] 点云减少比例: {(1 - downsampled_points.shape[0] / points.shape[0]) * 100:.1f}%")
    
    # 创建新的点云对象
    pruned_pcd = BasicPointCloud(
        points=downsampled_points,
        colors=downsampled_colors,
        normals=downsampled_normals
    )
    
    if motion_prob is not None:
        return pruned_pcd, downsampled_motion
    else:
        return pruned_pcd

