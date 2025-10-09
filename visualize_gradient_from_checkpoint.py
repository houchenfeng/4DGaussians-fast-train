#!/usr/bin/env python3
"""
从已保存的checkpoint生成梯度时间轴可视化

Usage:
    python visualize_gradient_from_checkpoint.py --model_path output/dynerf/sear_steak_test --iteration 14000
"""

import os
import sys
import torch
import argparse
from argparse import Namespace

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.gradient_tracker import GradientTracker


def load_model_and_scene(model_path, iteration):
    """
    加载模型和场景
    
    Args:
        model_path: 模型输出路径
        iteration: 迭代次数
        
    Returns:
        gaussians, scene, dataset, opt, pipe
    """
    print(f"Loading model from: {model_path}")
    print(f"Iteration: {iteration}")
    
    # 读取配置
    cfg_path = os.path.join(model_path, "cfg_args")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    with open(cfg_path, 'r') as f:
        cfg_content = f.read()
    
    # 解析配置
    args = eval(cfg_content.replace('Namespace(', 'dict(').replace(')', ')'))
    args = Namespace(**args)
    
    # 创建场景和模型
    print("Creating scene...")
    
    # 从args中提取参数
    class DatasetParams:
        def __init__(self, args):
            for key, value in vars(args).items():
                setattr(self, key, value)
    
    dataset = DatasetParams(args)
    dataset.model_path = model_path
    
    # 创建高斯模型
    print("Creating Gaussian model...")
    
    # ModelHiddenParams
    hyper_params = {
        'kplanes_config': getattr(args, 'kplanes_config', {}),
        'multires': getattr(args, 'multires', [1, 2, 4]),
        'defor_depth': getattr(args, 'defor_depth', 1),
        'net_width': getattr(args, 'net_width', 64),
        'plane_tv_weight': getattr(args, 'plane_tv_weight', 0.0002),
        'time_smoothness_weight': getattr(args, 'time_smoothness_weight', 0.01),
        'l1_time_planes': getattr(args, 'l1_time_planes', 0.0001),
        'no_do': getattr(args, 'no_do', False),
        'no_dshs': getattr(args, 'no_dshs', False),
        'no_ds': getattr(args, 'no_ds', False),
        'empty_voxel': getattr(args, 'empty_voxel', False),
        'static_mlp': getattr(args, 'static_mlp', False),
        'no_dr': getattr(args, 'no_dr', False),
        'no_dx': getattr(args, 'no_dx', False),
        'apply_rotation': getattr(args, 'apply_rotation', False),
        'no_grid': getattr(args, 'no_grid', False),
        'grid_pe': getattr(args, 'grid_pe', 0),
        'bounds': getattr(args, 'bounds', 1.6),
        'timebase_pe': getattr(args, 'timebase_pe', 4),
        'posebase_pe': getattr(args, 'posebase_pe', 10),
        'scale_rotation_pe': getattr(args, 'scale_rotation_pe', 2),
        'opacity_pe': getattr(args, 'opacity_pe', 2),
        'timenet_width': getattr(args, 'timenet_width', 64),
        'timenet_output': getattr(args, 'timenet_output', 32),
    }
    
    hyper = Namespace(**hyper_params)
    
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    
    # 加载场景
    print("Loading scene...")
    scene = Scene(dataset, gaussians, load_coarse=None, shuffle=False)
    
    # 加载checkpoint
    checkpoint_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    deformation_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # 加载PLY文件
    gaussians.load_ply(checkpoint_path)
    
    # 加载变形网络
    if os.path.exists(os.path.join(deformation_path, "deformation.pth")):
        print(f"Loading deformation network: {deformation_path}")
        gaussians.load_model(deformation_path)
    else:
        print("Warning: Deformation network not found, using default initialization")
    
    # 创建优化器（用于梯度计算）
    opt_params = {
        'iterations': getattr(args, 'iterations', 30000),
        'position_lr_init': getattr(args, 'position_lr_init', 0.00016),
        'position_lr_final': getattr(args, 'position_lr_final', 0.0000016),
        'position_lr_delay_mult': getattr(args, 'position_lr_delay_mult', 0.01),
        'position_lr_max_steps': getattr(args, 'position_lr_max_steps', 30000),
        'deformation_lr_init': getattr(args, 'deformation_lr_init', 0.00016),
        'deformation_lr_final': getattr(args, 'deformation_lr_final', 0.0000016),
        'deformation_lr_delay_mult': getattr(args, 'deformation_lr_delay_mult', 0.01),
        'grid_lr_init': getattr(args, 'grid_lr_init', 0.0016),
        'grid_lr_final': getattr(args, 'grid_lr_final', 0.000016),
        'feature_lr': getattr(args, 'feature_lr', 0.0025),
        'opacity_lr': getattr(args, 'opacity_lr', 0.05),
        'scaling_lr': getattr(args, 'scaling_lr', 0.005),
        'rotation_lr': getattr(args, 'rotation_lr', 0.001),
        'percent_dense': getattr(args, 'percent_dense', 0.01),
        'lambda_dssim': getattr(args, 'lambda_dssim', 0.2),
    }
    opt = Namespace(**opt_params)
    
    gaussians.training_setup(opt)
    
    # Pipeline参数
    pipe_params = {
        'convert_SHs_python': False,
        'compute_cov3D_python': False,
        'debug': False,
    }
    pipe = Namespace(**pipe_params)
    
    print(f"Model loaded successfully!")
    print(f"  Total points: {gaussians.get_xyz.shape[0]}")
    print(f"  Dataset type: {scene.dataset_type}")
    
    return gaussians, scene, dataset, opt, pipe


def main():
    parser = argparse.ArgumentParser(description="Visualize gradients from saved checkpoint")
    parser.add_argument("--model_path", type=str, 
                       default="/home/lt/2024/fast4dgs/4DGaussians-fast-train/output/dynerf/sear_steak_test",
                       help="Path to model output directory")
    parser.add_argument("--iteration", type=int, default=14000,
                       help="Iteration number to load")
    parser.add_argument("--time_points", nargs="+", type=float, 
                       default=None,
                       help="Time points to visualize (default: 0 0.1 0.2 ... 0.9)")
    parser.add_argument("--max_points", type=int, default=10000,
                       help="Maximum points to visualize")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: model_path/gradvis/iteration_X)")
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, f"gradvis/iteration_{args.iteration}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("Gradient Visualization from Checkpoint")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"Output dir: {args.output_dir}")
    print("")
    
    # Load model and scene
    try:
        gaussians, scene, dataset, opt, pipe = load_model_and_scene(
            args.model_path, args.iteration
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Set background
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Create gradient tracker
    gradient_tracker = GradientTracker(output_dir=args.output_dir, enable=True)
    
    # Set default time points
    if args.time_points is None:
        args.time_points = [i * 0.1 for i in range(10)]  # 0, 0.1, ..., 0.9
    
    print(f"\nTime points to visualize: {args.time_points}")
    print("")
    
    # Generate timeline visualization
    try:
        gradient_tracker.visualize_gradient_timeline(
            gaussians, scene, pipe, background,
            time_points=args.time_points,
            max_points=args.max_points
        )
    except Exception as e:
        print(f"\nError generating visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*60)
    print("✓ Visualization Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {os.path.join(args.output_dir, 'gradient_3d/gradient_timeline.html')}")
    print(f"\nOpen in browser:")
    print(f"  firefox {os.path.join(args.output_dir, 'gradient_3d/gradient_timeline.html')}")
    print(f"\nInteractive Controls:")
    print(f"  1. Point Size: Top-left dropdown (0.1x - 5x)")
    print(f"  2. Arrow Size: Second dropdown (0.5x - 16x)")
    print(f"  3. Visibility: Third dropdown (Both/Points/Arrows)")
    print(f"  4. Time Slider: Bottom slider to change time")
    print(f"  5. Play/Pause: Bottom-left buttons for animation")
    print(f"\nVisualization:")
    print(f"  • Arrow direction = Gradient direction (normalized)")
    print(f"  • Color (point & arrow) = Gradient magnitude (log scale)")
    print(f"  • Arrow length = Uniform (adjustable via dropdown)")
    print(f"\nNotes:")
    print(f"  • All settings persist during Play/Slider changes (JS-based)")
    print(f"  • Open browser console (F12) to see debug logs")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

