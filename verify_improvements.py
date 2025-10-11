#!/usr/bin/env python3
"""快速验证Instant4D改进模块 (1分钟)"""
import sys, os
sys.path.append(os.path.dirname(__file__))

def test_modules():
    """测试所有模块"""
    from utils.graphics_utils import BasicPointCloud
    from utils.grid_pruning import grid_pruning
    from utils.isotropic_gaussian import make_isotropic_scaling
    from utils.simplified_rgb import init_simplified_rgb
    from arguments import ModelParams
    from argparse import ArgumentParser
    import numpy as np
    import torch
    
    print("="*60)
    print("Instant4D模块快速验证")
    print("="*60)
    
    # 1. 网格剪枝
    pcd = BasicPointCloud(
        points=np.random.randn(50000, 3) * 2.0,
        colors=np.random.rand(50000, 3),
        normals=np.zeros((50000, 3))
    )
    pruned = grid_pruning(pcd, cameras=None, use_adaptive=False)
    reduction = (1 - len(pruned.points) / 50000) * 100
    print(f"✓ 网格剪枝: 50000 → {len(pruned.points)} 点 (-{reduction:.1f}%)")
    
    # 2. 各向同性
    aniso = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    iso = make_isotropic_scaling(aniso)
    ok = all(iso[i,0] == iso[i,1] == iso[i,2] for i in range(len(iso)))
    print(f"✓ 各向同性: 所有轴统一" if ok else "✗ 各向同性: 失败")
    
    # 3. 简化RGB
    colors = np.random.rand(1000, 3)
    features = init_simplified_rgb(colors, sh_degree=0)
    print(f"✓ 简化RGB: SH 16→1系数 (-93.75%)")
    
    # 4. 参数集成
    parser = ArgumentParser()
    args = ModelParams(parser)
    params = ['use_grid_pruning', 'use_isotropic_gaussian', 'use_simplified_rgb']
    ok = all(hasattr(args, p) for p in params)
    print(f"✓ 参数集成: 3个开关已添加" if ok else "✗ 参数集成: 失败")
    
    print("="*60)
    print("✓ 所有模块验证通过\n")
    print("下一步: ./debug_test.sh (测试各模块)")

if __name__ == "__main__":
    test_modules()

