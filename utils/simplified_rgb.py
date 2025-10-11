"""
简化RGB表示 - Instant4D 改进模块

解决问题: 原始4DGS使用复杂的球面谐波函数(SH)，参数数量庞大
改进方案: 使用简单的RGB值替代高阶SH

效果:
- 参数减少60%以上
- 计算加速
- 内存节省
- 减少过拟合
"""

import torch
import numpy as np


def simplify_sh_features(features, keep_degree=0):
    """
    简化球面谐波特征
    
    Args:
        features: (N, C, SH_coeffs) 原始特征
        keep_degree: 保留的SH阶数，0表示只保留DC分量(RGB)
        
    Returns:
        simplified_features: 简化后的特征
    """
    if keep_degree == 0:
        # 只保留DC分量（RGB值）
        features_dc = features[:, :, 0:1]
        # 将其他高阶项置零
        features[:, :, 1:] = 0.0
        print(f"[Simplified RGB] 简化SH特征: 只保留DC分量(RGB)")
    else:
        # 保留到指定阶数
        max_coeffs = (keep_degree + 1) ** 2
        features[:, :, max_coeffs:] = 0.0
        print(f"[Simplified RGB] 简化SH特征: 保留到{keep_degree}阶")
    
    return features


def init_simplified_rgb(pcd_colors, sh_degree=0):
    """
    初始化简化的RGB特征
    
    Args:
        pcd_colors: (N, 3) 点云颜色
        sh_degree: SH阶数，0表示只用RGB
        
    Returns:
        features: (N, 3, SH_coeffs) 特征张量
    """
    from utils.sh_utils import RGB2SH
    
    if torch.is_tensor(pcd_colors):
        fused_color = RGB2SH(pcd_colors)
    else:
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd_colors)).float().cuda())
    
    # 创建特征张量
    num_coeffs = (sh_degree + 1) ** 2
    features = torch.zeros((fused_color.shape[0], 3, num_coeffs)).float().cuda()
    
    # 只设置DC分量（RGB）
    features[:, :3, 0] = fused_color
    # 高阶项保持为0
    
    print(f"[Simplified RGB] 初始化RGB特征: {fused_color.shape[0]} 个点")
    print(f"[Simplified RGB] SH阶数: {sh_degree} (参数减少: {(1 - num_coeffs / ((3 + 1) ** 2)) * 100:.1f}%)")
    
    return features


def disable_sh_training(gaussian_model):
    """
    禁用高阶SH的训练
    
    Args:
        gaussian_model: GaussianModel实例
    """
    # 冻结高阶SH参数
    if hasattr(gaussian_model, '_features_rest'):
        gaussian_model._features_rest.requires_grad = False
        print("[Simplified RGB] 已禁用高阶SH训练")
        print("[Simplified RGB] 只训练DC分量(RGB)")


def configure_simplified_rgb(model_params):
    """
    配置简化RGB参数
    
    Args:
        model_params: 模型参数对象
    """
    # 限制SH阶数为0
    if hasattr(model_params, 'sh_degree'):
        original_degree = model_params.sh_degree
        model_params.sh_degree = 0
        print(f"[Simplified RGB] SH阶数: {original_degree} -> 0")
        print(f"[Simplified RGB] 参数减少: {(1 - 1 / ((original_degree + 1) ** 2)) * 100:.1f}%")

