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

# 设置matplotlib使用默认字体，避免字体警告
import matplotlib.font_manager
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

# 抑制matplotlib字体警告
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: Plotly not installed, 3D visualization will be disabled")


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
        os.makedirs(os.path.join(self.gradient_dir, 'gradient_3d'), exist_ok=True)
        
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
        
        # 保存最后一次的梯度快照（用于3D可视化）
        self.last_xyz_grad = None
        self.last_xyz = None
        self.last_iteration = 0
        self.last_stage = 'unknown'
        
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
            
            # 保存最后一次的xyz和梯度快照（用于3D可视化）
            self.last_xyz = gaussians._xyz.detach().clone()
            self.last_xyz_grad = xyz_grad.detach().clone()
            self.last_iteration = iteration
            self.last_stage = stage
        
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
    
    def visualize_gradient_3d(self, gaussians, stage='final', max_points=2000):
        """
        Generate interactive 3D HTML visualization of gradient magnitude and direction
        
        Args:
            gaussians: GaussianModel instance
            stage: 'coarse', 'fine', or 'final'
            max_points: Maximum number of points to visualize (downsampling)
        """
        if not self.enable or not HAS_PLOTLY:
            if not HAS_PLOTLY:
                print("Plotly not available, skipping 3D gradient visualization")
            return
        
        print(f"\nGenerating 3D gradient visualization ({stage})...")
        
        # Use saved gradient snapshot instead of current grad (which may be cleared)
        if self.last_xyz_grad is None:
            print("No saved gradient snapshot, skipping 3D visualization")
            print("  (Make sure gradient recording was enabled during training)")
            return
        
        xyz = self.last_xyz.cpu().numpy()  # [N, 3]
        grad = self.last_xyz_grad.cpu().numpy()  # [N, 3]
        grad_norm = np.linalg.norm(grad, axis=1)  # [N]
        
        # Filter out zero gradients
        nonzero_mask = grad_norm > 1e-10
        xyz_filtered = xyz[nonzero_mask]
        grad_filtered = grad[nonzero_mask]
        grad_norm_filtered = grad_norm[nonzero_mask]
        
        print(f"  Total points: {len(xyz)}, Non-zero gradients: {len(xyz_filtered)}")
        
        if len(xyz_filtered) == 0:
            print("  No non-zero gradients to visualize")
            return
        
        # Downsample if too many points - 点云和箭头使用相同的采样
        if len(xyz_filtered) > max_points:
            # Sample points with highest gradients (top 70%) + random (bottom 30%)
            sorted_indices = np.argsort(grad_norm_filtered)[::-1]
            top_k = int(max_points * 0.7)
            top_indices = sorted_indices[:top_k]
            
            # Random sample from remaining
            remaining_indices = sorted_indices[top_k:]
            random_count = max_points - top_k
            if random_count > 0 and len(remaining_indices) > 0:
                random_indices = np.random.choice(remaining_indices, 
                                                min(random_count, len(remaining_indices)), 
                                                replace=False)
                selected_indices = np.concatenate([top_indices, random_indices])
            else:
                selected_indices = top_indices
            
            xyz_filtered = xyz_filtered[selected_indices]
            grad_filtered = grad_filtered[selected_indices]
            grad_norm_filtered = grad_norm_filtered[selected_indices]
            
            print(f"  Downsampled to {len(xyz_filtered)} points")
        
        # 点云和箭头数量一致
        num_points = len(xyz_filtered)
        
        # Normalize gradient magnitude for color (log scale)
        grad_norm_log = np.log10(grad_norm_filtered + 1e-10)
        
        # Scale arrows for visibility
        scene_scale = np.max(xyz_filtered.max(axis=0) - xyz_filtered.min(axis=0))
        arrow_scale = scene_scale * 0.02  # 2% of scene size
        
        # Normalize gradient direction
        grad_dir = grad_filtered / (np.linalg.norm(grad_filtered, axis=1, keepdims=True) + 1e-10)
        grad_scaled = grad_dir * arrow_scale
        
        # Create Plotly figure with sliders
        fig = go.Figure()
        
        # Add scatter points colored by gradient magnitude
        fig.add_trace(go.Scatter3d(
            x=xyz_filtered[:, 0],
            y=xyz_filtered[:, 1],
            z=xyz_filtered[:, 2],
            mode='markers',
            marker=dict(
                size=3,  # Default size, can be adjusted
                color=grad_norm_log,
                colorscale='Hot',
                colorbar=dict(title="log10(Gradient Norm)"),
                showscale=True,
                sizemode='diameter'
            ),
            name='Points',
            text=[f'Grad: {g:.2e}' for g in grad_norm_filtered],
            hovertemplate='<b>Position:</b> (%{x:.3f}, %{y:.3f}, %{z:.3f})<br>' +
                         '<b>%{text}</b><br>' +
                         '<extra></extra>',
            visible=True
        ))
        
        # Add gradient direction arrows (与点云数量一致)
        fig.add_trace(go.Cone(
            x=xyz_filtered[:, 0],
            y=xyz_filtered[:, 1],
            z=xyz_filtered[:, 2],
            u=grad_scaled[:, 0],
            v=grad_scaled[:, 1],
            w=grad_scaled[:, 2],
            colorscale='Hot',
            sizemode='absolute',
            sizeref=arrow_scale * 0.5,
            showscale=False,
            name='Arrows',
            hovertemplate='<b>Gradient Direction</b><br>' +
                         'Magnitude: %{text}<br>' +
                         '<extra></extra>',
            text=[f'{g:.2e}' for g in grad_norm_filtered],
            visible=True
        ))
        
        # Add interactive controls with updatemenus
        fig.update_layout(
            title=dict(
                text=f'3D Gradient Visualization<br>' +
                     f'<sub>Captured at: Iteration {self.last_iteration} ({self.last_stage.upper()} stage) | ' +
                     f'Points/Arrows: {num_points} | ' +
                     f'After backward() before optimizer.step()</sub>',
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1400,
            height=900,
            hovermode='closest',
            updatemenus=[
                # Point size control
                dict(
                    buttons=[
                        dict(label="Point: 0.1",
                             method="restyle",
                             args=[{"marker.size": 0.1}, [0]]),
                        dict(label="Point: 0.5",
                             method="restyle",
                             args=[{"marker.size": 0.5}, [0]]),
                        dict(label="Point: 1",
                             method="restyle",
                             args=[{"marker.size": 1}, [0]]),
                        dict(label="Point: 2",
                             method="restyle",
                             args=[{"marker.size": 2}, [0]]),
                        dict(label="Point: 3",
                             method="restyle",
                             args=[{"marker.size": 3}, [0]]),
                        dict(label="Point: 5",
                             method="restyle",
                             args=[{"marker.size": 5}, [0]]),
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.02,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                ),
                # Arrow scale control
                dict(
                    buttons=[
                        dict(label="Arrow: 0.5x",
                             method="restyle",
                             args=[{"sizeref": arrow_scale * 0.5}, [1]]),
                        dict(label="Arrow: 1x",
                             method="restyle",
                             args=[{"sizeref": arrow_scale * 1.0}, [1]]),
                        dict(label="Arrow: 2x",
                             method="restyle",
                             args=[{"sizeref": arrow_scale * 2.0}, [1]]),
                        dict(label="Arrow: 4x",
                             method="restyle",
                             args=[{"sizeref": arrow_scale * 4.0}, [1]]),
                        dict(label="Arrow: 8x",
                             method="restyle",
                             args=[{"sizeref": arrow_scale * 8.0}, [1]]),
                        dict(label="Arrow: 16x",
                             method="restyle",
                             args=[{"sizeref": arrow_scale * 16.0}, [1]]),
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.18,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                ),
                # Toggle visibility
                dict(
                    buttons=[
                        dict(label="Show Both",
                             method="update",
                             args=[{"visible": [True, True]}]),
                        dict(label="Points Only",
                             method="update",
                             args=[{"visible": [True, False]}]),
                        dict(label="Arrows Only",
                             method="update",
                             args=[{"visible": [False, True]}]),
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.38,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                ),
            ]
        )
        
        # Save as interactive HTML
        html_path = os.path.join(self.gradient_dir, 'gradient_3d', 
                                f'gradient_3d_{stage}.html')
        fig.write_html(html_path)
        print(f"  Saved interactive HTML: {html_path}")
        print(f"  Open in browser: firefox {html_path}")
        
        # Save gradient statistics
        stats = {
            'visualization_stage': stage,
            'gradient_iteration': self.last_iteration,
            'gradient_stage': self.last_stage,
            'total_points': int(len(xyz)),
            'nonzero_gradients': int(len(xyz_filtered)),
            'visualized_points': num_points,
            'gradient_magnitude': {
                'min': float(grad_norm_filtered.min()),
                'max': float(grad_norm_filtered.max()),
                'mean': float(grad_norm_filtered.mean()),
                'median': float(np.median(grad_norm_filtered)),
                'std': float(grad_norm_filtered.std())
            },
            'note': f'Gradient captured at iteration {self.last_iteration} ({self.last_stage} stage), after backward() before optimizer.step()'
        }
        
        stats_path = os.path.join(self.gradient_dir, 'gradient_3d', 
                                 f'gradient_stats_{stage}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"  Gradient statistics:")
        print(f"    Mean: {stats['gradient_magnitude']['mean']:.6e}")
        print(f"    Max: {stats['gradient_magnitude']['max']:.6e}")
        print(f"    Min: {stats['gradient_magnitude']['min']:.6e}")
    
    def visualize_gradient_3d_snapshot(self, gaussians, iteration, stage, max_points=1000):
        """
        Generate 3D gradient snapshot during training (called periodically)
        
        Args:
            gaussians: GaussianModel instance (with current gradients)
            iteration: Current iteration number
            stage: Current stage ('coarse' or 'fine')
            max_points: Maximum points for fast visualization
        """
        if not self.enable or not HAS_PLOTLY:
            return
        
        # Get XYZ positions and gradients (from current state, not snapshot)
        if gaussians._xyz.grad is None:
            return
        
        xyz = gaussians._xyz.detach().cpu().numpy()  # [N, 3]
        grad = gaussians._xyz.grad.detach().cpu().numpy()  # [N, 3]
        grad_norm = np.linalg.norm(grad, axis=1)  # [N]
        
        # Filter out zero gradients
        nonzero_mask = grad_norm > 1e-10
        xyz_filtered = xyz[nonzero_mask]
        grad_filtered = grad[nonzero_mask]
        grad_norm_filtered = grad_norm[nonzero_mask]
        
        if len(xyz_filtered) == 0:
            return
        
        # Downsample for fast visualization
        if len(xyz_filtered) > max_points:
            sorted_indices = np.argsort(grad_norm_filtered)[::-1]
            top_k = int(max_points * 0.8)  # 80% highest gradients
            top_indices = sorted_indices[:top_k]
            
            remaining_indices = sorted_indices[top_k:]
            random_count = max_points - top_k
            if random_count > 0 and len(remaining_indices) > 0:
                random_indices = np.random.choice(remaining_indices, 
                                                min(random_count, len(remaining_indices)), 
                                                replace=False)
                selected_indices = np.concatenate([top_indices, random_indices])
            else:
                selected_indices = top_indices
            
            xyz_filtered = xyz_filtered[selected_indices]
            grad_filtered = grad_filtered[selected_indices]
            grad_norm_filtered = grad_norm_filtered[selected_indices]
        
        num_points = len(xyz_filtered)
        
        # Normalize gradient magnitude for color (log scale)
        grad_norm_log = np.log10(grad_norm_filtered + 1e-10)
        
        # Scale arrows
        scene_scale = np.max(xyz_filtered.max(axis=0) - xyz_filtered.min(axis=0))
        arrow_scale = scene_scale * 0.02
        
        grad_dir = grad_filtered / (np.linalg.norm(grad_filtered, axis=1, keepdims=True) + 1e-10)
        grad_scaled = grad_dir * arrow_scale
        
        # Create figure
        fig = go.Figure()
        
        # Add points
        fig.add_trace(go.Scatter3d(
            x=xyz_filtered[:, 0],
            y=xyz_filtered[:, 1],
            z=xyz_filtered[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=grad_norm_log,
                colorscale='Hot',
                colorbar=dict(title="log10(Grad)"),
                showscale=True
            ),
            name='Points',
            text=[f'{g:.2e}' for g in grad_norm_filtered],
            hovertemplate='Pos: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>Grad: %{text}<extra></extra>',
            visible=True
        ))
        
        # Add arrows
        fig.add_trace(go.Cone(
            x=xyz_filtered[:, 0],
            y=xyz_filtered[:, 1],
            z=xyz_filtered[:, 2],
            u=grad_scaled[:, 0],
            v=grad_scaled[:, 1],
            w=grad_scaled[:, 2],
            colorscale='Hot',
            sizemode='absolute',
            sizeref=arrow_scale * 0.5,
            showscale=False,
            name='Arrows',
            text=[f'{g:.2e}' for g in grad_norm_filtered],
            hovertemplate='Grad: %{text}<extra></extra>',
            visible=True
        ))
        
        # Layout with controls
        fig.update_layout(
            title=f'Gradient Snapshot - Iter {iteration} ({stage.upper()})<br>' +
                  f'<sub>Points/Arrows: {num_points}</sub>',
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data'
            ),
            width=1200, height=800,
            updatemenus=[
                dict(
                    buttons=[
                        dict(label="Point: 0.1", method="restyle", args=[{"marker.size": 0.1}, [0]]),
                        dict(label="Point: 0.5", method="restyle", args=[{"marker.size": 0.5}, [0]]),
                        dict(label="Point: 1", method="restyle", args=[{"marker.size": 1}, [0]]),
                        dict(label="Point: 2", method="restyle", args=[{"marker.size": 2}, [0]]),
                        dict(label="Point: 3", method="restyle", args=[{"marker.size": 3}, [0]]),
                        dict(label="Point: 5", method="restyle", args=[{"marker.size": 5}, [0]]),
                    ],
                    direction="down", showactive=True,
                    x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="white", bordercolor="gray", borderwidth=1
                ),
                dict(
                    buttons=[
                        dict(label="Arrow: 0.5x", method="restyle", args=[{"sizeref": arrow_scale*0.5}, [1]]),
                        dict(label="Arrow: 1x", method="restyle", args=[{"sizeref": arrow_scale*1.0}, [1]]),
                        dict(label="Arrow: 2x", method="restyle", args=[{"sizeref": arrow_scale*2.0}, [1]]),
                        dict(label="Arrow: 4x", method="restyle", args=[{"sizeref": arrow_scale*4.0}, [1]]),
                        dict(label="Arrow: 8x", method="restyle", args=[{"sizeref": arrow_scale*8.0}, [1]]),
                        dict(label="Arrow: 16x", method="restyle", args=[{"sizeref": arrow_scale*16.0}, [1]]),
                    ],
                    direction="down", showactive=True,
                    x=0.18, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="white", bordercolor="gray", borderwidth=1
                ),
                dict(
                    buttons=[
                        dict(label="Both", method="update", args=[{"visible": [True, True]}]),
                        dict(label="Points", method="update", args=[{"visible": [True, False]}]),
                        dict(label="Arrows", method="update", args=[{"visible": [False, True]}]),
                    ],
                    direction="down", showactive=True,
                    x=0.36, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="white", bordercolor="gray", borderwidth=1
                ),
            ]
        )
        
        # Save HTML
        html_path = os.path.join(self.gradient_dir, 'gradient_3d', 
                                f'grad3d_{stage}_iter_{iteration:06d}.html')
        fig.write_html(html_path)
        print(f"  Saved 3D gradient: {html_path}")

