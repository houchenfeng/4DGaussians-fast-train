"""
各向同性高斯 (Isotropic Gaussian) - Instant4D 改进模块

解决问题: 原始4DGS使用各向异性高斯，在单目场景中容易导致数值不稳定
改进方案: 使用各向同性高斯（所有轴使用相同的缩放值）

效果: 
- 提高数值稳定性
- 减少参数复杂度
- 更快收敛
- PSNR提升 1.25dB
"""

import torch


def make_isotropic_scaling(scaling_params):
    """
    将各向异性缩放转换为各向同性缩放
    
    使用第一个缩放参数应用到所有三个轴
    
    Args:
        scaling_params: (N, 3) 的缩放参数
        
    Returns:
        isotropic_scaling: (N, 3) 各向同性缩放（所有轴相同）
    """
    if scaling_params.shape[-1] >= 3:
        # 使用第一个参数，重复3次
        iso_scale = scaling_params[:, 0:1].repeat(1, 3)
        return iso_scale
    else:
        return scaling_params


def apply_isotropic_gaussian(gaussian_model):
    """
    将高斯模型转换为各向同性
    
    Args:
        gaussian_model: GaussianModel实例
    """
    # 修改 get_scaling 方法
    original_get_scaling = gaussian_model.get_scaling
    
    @property
    def isotropic_get_scaling(self):
        anisotropic_scaling = original_get_scaling.fget(self)
        return make_isotropic_scaling(anisotropic_scaling)
    
    # 替换方法
    type(gaussian_model).get_scaling = isotropic_get_scaling
    
    print("[Isotropic Gaussian] 已启用各向同性高斯")
    print("[Isotropic Gaussian] 使用第一个缩放参数应用到所有轴")


class IsotropicGaussianWrapper:
    """各向同性高斯包装器"""
    
    def __init__(self, base_model):
        self.base_model = base_model
        
    @property  
    def get_scaling(self):
        """获取各向同性缩放"""
        base_scaling = self.base_model.scaling_activation(self.base_model._scaling)
        # 使用第一个参数重复3次
        return base_scaling[:, 0:1].repeat(1, 3)
    
    def __getattr__(self, name):
        """代理其他属性到基础模型"""
        return getattr(self.base_model, name)

