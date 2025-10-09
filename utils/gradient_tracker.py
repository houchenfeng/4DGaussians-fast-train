"""
梯度追踪模块
用于记录训练过程中的梯度信息并生成可视化
"""

import torch
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


class GradientTracker:
    """追踪训练过程中的梯度信息"""
    
    def __init__(self, output_dir, enable=True):
        """
        初始化梯度追踪器
        
        Args:
            output_dir: 输出目录
            enable: 是否启用追踪
        """
        self.output_dir = output_dir
        self.enable = enable
        
        if not self.enable:
            return
            
        self.gradient_dir = os.path.join(output_dir, 'gradient_vis')
        os.makedirs(self.gradient_dir, exist_ok=True)
        os.makedirs(os.path.join(self.gradient_dir, 'gradient_curves'), exist_ok=True)
        os.makedirs(os.path.join(self.gradient_dir, 'gradient_heatmaps'), exist_ok=True)
        
        # 梯度历史记录 - 分别存储coarse和fine阶段
        self.gradient_history = {
            'coarse': {
                'iterations': [],
                'xyz': [],
                'opacity': [],
                'scale': [],
                'rotation': [],
                'features_dc': [],
                'features_rest': [],
                'viewspace': [],
                'deformation_mlp': [],
                'deformation_grid': [],
            },
            'fine': {
                'iterations': [],
                'xyz': [],
                'opacity': [],
                'scale': [],
                'rotation': [],
                'features_dc': [],
                'features_rest': [],
                'viewspace': [],
                'deformation_mlp': [],
                'deformation_grid': [],
            }
        }
        
        # 详细的梯度统计
        self.detailed_stats = []
        
    def record_gradients(self, gaussians, iteration, viewspace_point_tensor_grad=None, stage='fine'):
        """
        记录当前iteration的梯度
        
        Args:
            gaussians: GaussianModel实例
            iteration: 当前迭代次数
            viewspace_point_tensor_grad: 屏幕空间梯度
            stage: 训练阶段 ('coarse' or 'fine')
        """
        if not self.enable:
            return
            
        stats = {
            'iteration': iteration,
            'stage': stage,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 记录高斯参数的梯度
        stats['gradients'] = {}
        
        # XYZ梯度
        if gaussians._xyz.grad is not None:
            xyz_grad = gaussians._xyz.grad
            stats['gradients']['xyz'] = self._compute_gradient_stats(xyz_grad, 'xyz')
        
        # Opacity梯度
        if gaussians._opacity.grad is not None:
            opacity_grad = gaussians._opacity.grad
            stats['gradients']['opacity'] = self._compute_gradient_stats(opacity_grad, 'opacity')
        
        # Scale梯度
        if gaussians._scaling.grad is not None:
            scale_grad = gaussians._scaling.grad
            stats['gradients']['scale'] = self._compute_gradient_stats(scale_grad, 'scale')
        
        # Rotation梯度
        if gaussians._rotation.grad is not None:
            rotation_grad = gaussians._rotation.grad
            stats['gradients']['rotation'] = self._compute_gradient_stats(rotation_grad, 'rotation')
        
        # Features梯度
        if gaussians._features_dc.grad is not None:
            features_dc_grad = gaussians._features_dc.grad
            stats['gradients']['features_dc'] = self._compute_gradient_stats(features_dc_grad, 'features_dc')
        
        if gaussians._features_rest.grad is not None:
            features_rest_grad = gaussians._features_rest.grad
            stats['gradients']['features_rest'] = self._compute_gradient_stats(features_rest_grad, 'features_rest')
        
        # Viewspace梯度
        if viewspace_point_tensor_grad is not None:
            stats['gradients']['viewspace'] = self._compute_gradient_stats(viewspace_point_tensor_grad, 'viewspace')
        
        # 变形网络梯度
        deformation_mlp_grads = []
        deformation_grid_grads = []
        
        for name, param in gaussians._deformation.named_parameters():
            if param.grad is not None:
                grad_stats = self._compute_gradient_stats(param.grad, name)
                grad_stats['param_name'] = name
                
                if 'grid' in name:
                    deformation_grid_grads.append(grad_stats)
                else:
                    deformation_mlp_grads.append(grad_stats)
        
        stats['deformation_mlp'] = deformation_mlp_grads
        stats['deformation_grid'] = deformation_grid_grads
        
        # 保存统计信息
        self.detailed_stats.append(stats)
        
        # 更新历史记录（用于绘图）- 根据stage分别存储
        stage_history = self.gradient_history[stage]
        stage_history['iterations'].append(iteration)
        for key in ['xyz', 'opacity', 'scale', 'rotation', 'features_dc', 'features_rest', 'viewspace']:
            if key in stats['gradients']:
                stage_history[key].append(stats['gradients'][key]['norm'])
            else:
                stage_history[key].append(0.0)
        
        # MLP和Grid的平均梯度范数
        if deformation_mlp_grads:
            mlp_norm = np.mean([g['norm'] for g in deformation_mlp_grads])
            stage_history['deformation_mlp'].append(mlp_norm)
        else:
            stage_history['deformation_mlp'].append(0.0)
        
        if deformation_grid_grads:
            grid_norm = np.mean([g['norm'] for g in deformation_grid_grads])
            stage_history['deformation_grid'].append(grid_norm)
        else:
            stage_history['deformation_grid'].append(0.0)
    
    def _compute_gradient_stats(self, grad_tensor, name):
        """
        计算梯度张量的统计信息
        
        Args:
            grad_tensor: 梯度张量
            name: 参数名称
            
        Returns:
            dict: 梯度统计信息
        """
        grad_abs = grad_tensor.abs()
        
        stats = {
            'mean': grad_abs.mean().item(),
            'std': grad_tensor.std().item(),
            'max': grad_abs.max().item(),
            'min': grad_abs.min().item(),
            'norm': torch.norm(grad_tensor).item(),
            'shape': list(grad_tensor.shape),
        }
        
        # 检测梯度异常
        stats['has_nan'] = torch.isnan(grad_tensor).any().item()
        stats['has_inf'] = torch.isinf(grad_tensor).any().item()
        
        # 梯度消失/爆炸检测
        if stats['norm'] < 1e-7:
            stats['warning'] = 'gradient_vanishing'
        elif stats['norm'] > 1e3:
            stats['warning'] = 'gradient_explosion'
        else:
            stats['warning'] = None
        
        return stats
    
    def visualize_gradient_curves(self, iteration=None, save=True):
        """
        可视化梯度曲线 - 为coarse和fine阶段分别生成独立的图
        
        Args:
            iteration: 当前迭代次数（用于文件名）
            save: 是否保存图像
        """
        if not self.enable:
            return
        
        # 检查是否有数据
        has_coarse = len(self.gradient_history['coarse']['iterations']) > 0
        has_fine = len(self.gradient_history['fine']['iterations']) > 0
        
        if not has_coarse and not has_fine:
            return
        
        # 定义要绘制的梯度类型
        grad_types = [
            ('xyz', 'XYZ Position Gradient'),
            ('opacity', 'Opacity Gradient'),
            ('scale', 'Scale Gradient'),
            ('rotation', 'Rotation Gradient'),
            ('features_dc', 'Features DC Gradient'),
            ('features_rest', 'Features Rest Gradient'),
            ('viewspace', 'Viewspace Gradient'),
            ('deformation_mlp', 'Deformation MLP Gradient'),
            ('deformation_grid', 'Deformation Grid Gradient'),
        ]
        
        # 为每个阶段生成独立的图
        for stage in ['coarse', 'fine']:
            if stage == 'coarse' and not has_coarse:
                continue
            if stage == 'fine' and not has_fine:
                continue
            
            # 创建图
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            fig.suptitle(f'Gradient Norm Curves - {stage.upper()} Stage', fontsize=16)
            
            iterations = self.gradient_history[stage]['iterations']
            
            for idx, (key, title) in enumerate(grad_types):
                ax = axes[idx // 3, idx % 3]
                values = self.gradient_history[stage][key]
                
                # 只绘制非零值
                if any(v > 0 for v in values):
                    valid_indices = [i for i, v in enumerate(values) if v > 0]
                    if valid_indices:
                        valid_iterations = [iterations[i] for i in valid_indices]
                        valid_values = [values[i] for i in valid_indices]
                        ax.plot(valid_iterations, valid_values, linewidth=2, marker='o', markersize=3)
                        ax.set_yscale('log')
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Gradient Norm (log scale)')
                        ax.set_title(title)
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, 'No gradient data', 
                               ha='center', va='center',
                               transform=ax.transAxes)
                        ax.set_title(title)
                else:
                    ax.text(0.5, 0.5, 'No gradient data', 
                           ha='center', va='center',
                           transform=ax.transAxes)
                    ax.set_title(title)
            
            plt.tight_layout()
            
            if save:
                if iteration is not None:
                    filename = f'gradient_curves_{stage}_iter_{iteration:06d}.png'
                else:
                    filename = f'gradient_curves_{stage}_final.png'
                
                save_path = os.path.join(self.gradient_dir, 'gradient_curves', filename)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Gradient curves ({stage}) saved to: {save_path}")
            
            plt.close(fig)
    
    def save_gradient_stats(self, filename='gradient_stats.json'):
        """Save gradient statistics to JSON file"""
        if not self.enable:
            return
        
        save_path = os.path.join(self.gradient_dir, filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'gradient_history': self.gradient_history,
                'detailed_stats': self.detailed_stats
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Gradient stats saved to: {save_path}")
    
    def generate_report(self):
        """Generate gradient analysis report"""
        if not self.enable:
            return
        
        print("\n" + "="*60)
        print("Gradient Analysis Report")
        print("="*60)
        
        if len(self.detailed_stats) == 0:
            print("No gradient data")
            return
        
        # Statistics of last iteration
        final_stats = self.detailed_stats[-1]
        print(f"\nLast iteration: {final_stats['iteration']}")
        print(f"Timestamp: {final_stats['timestamp']}")
        
        print("\nGradient Statistics:")
        for key, stats in final_stats['gradients'].items():
            print(f"\n{key}:")
            print(f"  Norm: {stats['norm']:.6e}")
            print(f"  Mean: {stats['mean']:.6e}")
            print(f"  Max: {stats['max']:.6e}")
            print(f"  Min: {stats['min']:.6e}")
            if stats.get('warning'):
                print(f"  ⚠️  Warning: {stats['warning']}")
        
        # Check anomalies during training
        print("\nAnomaly Detection:")
        warnings_count = {
            'gradient_vanishing': 0,
            'gradient_explosion': 0,
            'has_nan': 0,
            'has_inf': 0
        }
        
        for stats in self.detailed_stats:
            for grad_stats in stats['gradients'].values():
                if grad_stats.get('warning'):
                    warnings_count[grad_stats['warning']] += 1
                if grad_stats.get('has_nan'):
                    warnings_count['has_nan'] += 1
                if grad_stats.get('has_inf'):
                    warnings_count['has_inf'] += 1
        
        for warning_type, count in warnings_count.items():
            if count > 0:
                print(f"  {warning_type}: {count} times")
        
        # Save statistics and visualizations
        self.save_gradient_stats()
        self.visualize_gradient_curves()
        
        print("\n" + "="*60)

