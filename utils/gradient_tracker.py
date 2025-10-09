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
        
        # 采样策略: 80% 最大梯度 + 20% 随机采样
        if len(xyz_filtered) > max_points:
            sorted_indices = np.argsort(grad_norm_filtered)[::-1]
            
            # 80% 取最大梯度
            n_top = int(max_points * 0.8)
            top_indices = sorted_indices[:n_top]
            
            # 20% 随机采样（从剩余的点中）
            n_random = max_points - n_top
            remaining_indices = sorted_indices[n_top:]
            if len(remaining_indices) > n_random:
                random_indices = np.random.choice(remaining_indices, n_random, replace=False)
            else:
                random_indices = remaining_indices
            
            # 合并
            selected_indices = np.concatenate([top_indices, random_indices])
            
            xyz_filtered = xyz_filtered[selected_indices]
            grad_filtered = grad_filtered[selected_indices]
            grad_norm_filtered = grad_norm_filtered[selected_indices]
            
            print(f"  Stratified sampling to {len(xyz_filtered)} points (by gradient magnitude)")
        
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
        
        # 采样策略: 80% 最大梯度 + 20% 随机采样
        if len(xyz_filtered) > max_points:
            sorted_indices = np.argsort(grad_norm_filtered)[::-1]
            
            # 80% 取最大梯度
            n_top = int(max_points * 0.8)
            top_indices = sorted_indices[:n_top]
            
            # 20% 随机采样（从剩余的点中）
            n_random = max_points - n_top
            remaining_indices = sorted_indices[n_top:]
            if len(remaining_indices) > n_random:
                random_indices = np.random.choice(remaining_indices, n_random, replace=False)
            else:
                random_indices = remaining_indices
            
            # 合并
            selected_indices = np.concatenate([top_indices, random_indices])
            
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
    
    def visualize_gradient_timeline(self, gaussians, scene, pipe, background, 
                                   time_points=None, max_points=1000):
        """
        Visualize gradients at different time points with interactive slider
        
        Args:
            gaussians: GaussianModel instance
            scene: Scene instance
            pipe: Pipeline parameters
            background: Background color
            time_points: List of time points to sample (default: [0, 0.1, ..., 0.9])
            max_points: Maximum points to visualize
        """
        if not self.enable or not HAS_PLOTLY:
            if not HAS_PLOTLY:
                print("Plotly not available, skipping timeline visualization")
            return
        
        print("\n=== Generating Gradient Timeline Visualization ===")
        
        # Default time points
        if time_points is None:
            time_points = [i * 0.1 for i in range(10)]  # [0, 0.1, 0.2, ..., 0.9]
        
        print(f"Computing gradients at {len(time_points)} time points...")
        
        # Get a representative camera
        train_cams = scene.getTrainCameras()
        if len(train_cams) == 0:
            print("No training cameras available")
            return
        
        # Use first camera as reference
        viewpoint = train_cams[0]
        
        # Import render function
        from gaussian_renderer import render
        from utils.loss_utils import l1_loss
        
        # Store gradients and deformed positions for all time points
        all_gradients = []
        all_deformed_xyz = []
        all_losses = []
        
        # Get base xyz and parameters
        xyz_base = gaussians.get_xyz
        scales = gaussians._scaling
        rotations = gaussians._rotation
        opacity = gaussians._opacity
        shs = gaussians.get_features
        
        for t in time_points:
            print(f"  Computing gradient at t={t:.1f}...", end=" ")
            
            # Clear previous gradients
            gaussians.optimizer.zero_grad()
            
            # Set time
            if hasattr(viewpoint, 'time'):
                viewpoint.time = t
            
            # Forward pass
            render_pkg = render(viewpoint, gaussians, pipe, background, 
                              stage='fine', cam_type=scene.dataset_type)
            rendered = render_pkg["render"]
            
            # Get ground truth
            if scene.dataset_type != "PanopticSports":
                gt = viewpoint.original_image.cuda()
            else:
                gt = viewpoint['image'].cuda()
            
            # Compute loss
            loss = l1_loss(rendered, gt)
            
            # Backward pass
            loss.backward()
            
            # Save gradient and compute deformed position
            if gaussians._xyz.grad is not None:
                xyz_grad = gaussians._xyz.grad.detach().clone().cpu().numpy()
                
                # Compute deformed position at time t (without gradient tracking)
                with torch.no_grad():
                    time_t = torch.tensor(t).cuda().repeat(xyz_base.shape[0], 1)
                    xyz_deformed, _, _, _, _ = gaussians._deformation(
                        xyz_base, scales, rotations, opacity, shs, time_t
                    )
                    xyz_deformed_np = xyz_deformed.cpu().numpy()
                
                all_gradients.append(xyz_grad)
                all_deformed_xyz.append(xyz_deformed_np)
                all_losses.append(loss.item())
                print(f"Loss: {loss.item():.6f}, Grad norm: {np.linalg.norm(xyz_grad):.6e}")
            else:
                print("No gradient")
                all_gradients.append(None)
                all_deformed_xyz.append(None)
                all_losses.append(None)
            
            # Clear gradients
            gaussians.optimizer.zero_grad()
        
        # Filter out None gradients
        valid_indices = [i for i, g in enumerate(all_gradients) if g is not None]
        if len(valid_indices) == 0:
            print("No valid gradients computed")
            return
        
        print(f"\nCreating interactive timeline with {len(valid_indices)} time points...")
        
        # 统计变形量（调试信息）
        xyz_base_np = gaussians.get_xyz.detach().cpu().numpy()
        print(f"\nDeformation statistics (checking if scene is dynamic):")
        for idx in valid_indices:
            t = time_points[idx]
            xyz_def = all_deformed_xyz[idx]
            deformation = xyz_def - xyz_base_np
            deform_norm = np.linalg.norm(deformation, axis=1)
            print(f"  t={t:.1f}: Deform mean={deform_norm.mean():.6f}, max={deform_norm.max():.6f}, "
                  f"median={np.median(deform_norm):.6f}, nonzero={np.sum(deform_norm > 1e-6)}/{len(deform_norm)}")
        
        # 检查是否有显著变形
        max_deform = max([np.linalg.norm(all_deformed_xyz[idx] - xyz_base_np, axis=1).max() 
                         for idx in valid_indices])
        if max_deform < 0.001:
            print(f"\n⚠️  WARNING: Maximum deformation is very small ({max_deform:.6f})")
            print(f"  This might be a static scene or deformation network hasn't learned yet")
            print(f"  Scene maxtime: {scene.maxtime if hasattr(scene, 'maxtime') else 'unknown'}")
        
        # 计算所有时刻的统一显示范围（重要！）
        all_xyz_combined = []
        for idx in valid_indices:
            all_xyz_combined.append(all_deformed_xyz[idx])
        all_xyz_combined = np.concatenate(all_xyz_combined, axis=0)
        
        # 计算全局bbox（所有时刻的xyz范围）
        xyz_min = all_xyz_combined.min(axis=0)
        xyz_max = all_xyz_combined.max(axis=0)
        xyz_center = (xyz_min + xyz_max) / 2
        xyz_range = (xyz_max - xyz_min).max() * 0.6  # 使用最大范围的60%作为统一范围
        
        print(f"  Unified scene range: center={xyz_center}, range={xyz_range:.3f}")
        
        # 计算统一的箭头scale（使用全局scene范围）
        unified_arrow_scale = xyz_range * 0.02  # 所有时刻使用相同的箭头scale
        print(f"  Unified arrow scale: {unified_arrow_scale:.6f}")
        
        # 计算所有时刻的梯度范围（用于统一颜色映射）
        all_grad_norms = []
        for idx in valid_indices:
            grad = all_gradients[idx]
            grad_norm = np.linalg.norm(grad, axis=1)
            all_grad_norms.append(grad_norm[grad_norm > 1e-10])
        
        all_grad_norms = np.concatenate(all_grad_norms)
        grad_norm_log_min = np.log10(all_grad_norms.min() + 1e-10)
        grad_norm_log_max = np.log10(all_grad_norms.max() + 1e-10)
        
        print(f"  Unified color range: log10(grad) in [{grad_norm_log_min:.2f}, {grad_norm_log_max:.2f}]")
        
        # Create Plotly figure with frames
        frames = []
        
        for idx in valid_indices:
            t = time_points[idx]
            grad = all_gradients[idx]
            xyz_deformed = all_deformed_xyz[idx]  # 使用变形后的位置
            loss = all_losses[idx]
            
            grad_norm = np.linalg.norm(grad, axis=1)
            
            # Filter zero gradients
            nonzero_mask = grad_norm > 1e-10
            xyz_filtered = xyz_deformed[nonzero_mask]  # 使用变形后的位置
            grad_filtered = grad[nonzero_mask]
            grad_norm_filtered = grad_norm[nonzero_mask]
            
            # 采样策略: 80% 最大梯度 + 20% 随机采样
            if len(xyz_filtered) > max_points:
                sorted_indices = np.argsort(grad_norm_filtered)[::-1]
                
                # 80% 取最大梯度
                n_top = int(max_points * 0.8)
                top_indices = sorted_indices[:n_top]
                
                # 20% 随机采样（从剩余的点中）
                n_random = max_points - n_top
                remaining_indices = sorted_indices[n_top:]
                if len(remaining_indices) > n_random:
                    random_indices = np.random.choice(remaining_indices, n_random, replace=False)
                else:
                    random_indices = remaining_indices
                
                # 合并
                selected_indices = np.concatenate([top_indices, random_indices])
                
                xyz_filtered = xyz_filtered[selected_indices]
                grad_filtered = grad_filtered[selected_indices]
                grad_norm_filtered = grad_norm_filtered[selected_indices]
            
            # Color by gradient magnitude
            grad_norm_log = np.log10(grad_norm_filtered + 1e-10)
            
            # Arrow vectors - 统一长度，仅显示方向
            # 归一化梯度向量
            grad_norm_filtered_safe = grad_norm_filtered + 1e-10
            grad_dir = grad_filtered / grad_norm_filtered_safe[:, np.newaxis]
            
            # Create frame data
            frame_data = [
                go.Scatter3d(
                    x=xyz_filtered[:, 0],
                    y=xyz_filtered[:, 1],
                    z=xyz_filtered[:, 2],
                    mode='markers',
                    marker=dict(
                        color=grad_norm_log, 
                        colorscale='Hot',
                        sizemode='diameter',
                        cmin=grad_norm_log_min,  # 统一的颜色范围
                        cmax=grad_norm_log_max
                    ),
                    name='Points',
                    text=[f'{g:.2e}' for g in grad_norm_filtered],
                    hovertemplate='Pos: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>Grad: %{text}<extra></extra>'
                ),
                # 箭头：统一长度，方向表示梯度方向，颜色表示梯度大小
                go.Cone(
                    x=xyz_filtered[:, 0],
                    y=xyz_filtered[:, 1],
                    z=xyz_filtered[:, 2],
                    u=grad_dir[:, 0],  # 使用归一化梯度向量
                    v=grad_dir[:, 1],
                    w=grad_dir[:, 2],
                    colorscale='Hot',
                    sizemode='scaled',
                    sizeref=unified_arrow_scale * 0.5,  # 统一箭头大小
                    showscale=False,
                    name='Arrows'
                )
            ]
            
            # Create frame with annotation showing point count
            frame = go.Frame(
                data=frame_data,
                name=f't_{t:.1f}',
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=f't={t:.1f} | Loss={loss:.6f} | Points={len(xyz_filtered)}',
                            xref='paper', yref='paper',
                            x=0.5, y=1.05,
                            xanchor='center', yanchor='bottom',
                            showarrow=False,
                            font=dict(size=14, color='blue')
                        )
                    ]
                )
            )
            frames.append(frame)
        
        # Create initial figure with first frame
        first_idx = valid_indices[0]
        t_first = time_points[first_idx]
        loss_first = all_losses[first_idx]
        
        # Recompute first frame data for initial display
        grad_first = all_gradients[first_idx]
        xyz_deformed_first = all_deformed_xyz[first_idx]  # 使用变形后的位置
        grad_norm_first = np.linalg.norm(grad_first, axis=1)
        nonzero_mask = grad_norm_first > 1e-10
        xyz_init = xyz_deformed_first[nonzero_mask]  # 使用变形后的位置
        grad_init = grad_first[nonzero_mask]
        grad_norm_init = grad_norm_first[nonzero_mask]
        
        if len(xyz_init) > max_points:
            sorted_indices = np.argsort(grad_norm_init)[::-1]
            
            # 80% 取最大梯度
            n_top = int(max_points * 0.8)
            top_indices = sorted_indices[:n_top]
            
            # 20% 随机采样（从剩余的点中）
            n_random = max_points - n_top
            remaining_indices = sorted_indices[n_top:]
            if len(remaining_indices) > n_random:
                random_indices = np.random.choice(remaining_indices, n_random, replace=False)
            else:
                random_indices = remaining_indices
            
            # 合并
            selected_indices = np.concatenate([top_indices, random_indices])
            
            xyz_init = xyz_init[selected_indices]
            grad_init = grad_init[selected_indices]
            grad_norm_init = grad_norm_init[selected_indices]
        
        grad_norm_log_init = np.log10(grad_norm_init + 1e-10)
        
        # 归一化初始梯度
        grad_norm_init_safe = grad_norm_init + 1e-10
        grad_dir_init = grad_init / grad_norm_init_safe[:, np.newaxis]
        
        # 初始figure
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=xyz_init[:, 0], y=xyz_init[:, 1], z=xyz_init[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,  # 默认点大小
                        color=grad_norm_log_init, 
                        colorscale='Hot',
                        colorbar=dict(title="log10(Grad)", x=1.02),
                        sizemode='diameter',
                        cmin=grad_norm_log_min,  # 统一的颜色范围
                        cmax=grad_norm_log_max
                    ),
                    name='Points'
                ),
                # 箭头：统一长度，方向表示梯度方向，颜色表示梯度大小
                go.Cone(
                    x=xyz_init[:, 0], y=xyz_init[:, 1], z=xyz_init[:, 2],
                    u=grad_dir_init[:, 0],  # 使用归一化梯度向量
                    v=grad_dir_init[:, 1],
                    w=grad_dir_init[:, 2],
                    colorscale='Hot',
                    sizemode='scaled',
                    sizeref=unified_arrow_scale * 0.5,  # 统一箭头大小
                    showscale=False,
                    name='Arrows'
                )
            ],
            frames=frames
        )
        
        # Create slider
        sliders = [dict(
            active=0,
            yanchor="top",
            y=0.02,
            xanchor="left",
            x=0.05,
            currentvalue=dict(
                prefix="Time: t=",
                visible=True,
                xanchor="left"
            ),
            transition=dict(duration=300),
            pad=dict(b=10, t=50),
            len=0.9,
            steps=[dict(
                args=[[f.name], dict(
                    frame=dict(duration=300, redraw=True),
                    mode="immediate",
                    transition=dict(duration=300)
                )],
                label=f"{time_points[idx]:.1f}",
                method="animate"
            ) for idx, f in zip(valid_indices, frames)]
        )]
        
        # 计算初始帧的实际点数
        initial_point_count = len(xyz_init)
        
        # Update layout with unified scene range
        fig.update_layout(
            title=f'Gradient Timeline Visualization<br>' +
                  f'<sub>Point positions = xyz_base + Δxyz(t) | Arrows = gradient direction | Drag slider to change time</sub>',
            annotations=[
                dict(
                    text=f't={t_first:.1f} | Loss={loss_first:.6f} | Points={initial_point_count}',
                    xref='paper', yref='paper',
                    x=0.5, y=1.05,
                    xanchor='center', yanchor='bottom',
                    showarrow=False,
                    font=dict(size=14, color='blue')
                )
            ],
            scene=dict(
                xaxis_title='X', 
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='cube',  # 保持各轴比例一致
                # 设置统一的显示范围
                xaxis=dict(range=[xyz_center[0]-xyz_range, xyz_center[0]+xyz_range]),
                yaxis=dict(range=[xyz_center[1]-xyz_range, xyz_center[1]+xyz_range]),
                zaxis=dict(range=[xyz_center[2]-xyz_range, xyz_center[2]+xyz_range]),
                # 固定相机位置
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1),
                    projection=dict(type='perspective')
                )
            ),
            width=1400,
            height=900,
            sliders=sliders,
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
                    active=2,  # 默认选中 Point: 1
                    x=0.02,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                ),
                # Arrow size control (统一大小调整)
                dict(
                    buttons=[
                        dict(label="Arrow: 0.5x",
                             method="restyle",
                             args=[{"sizeref": unified_arrow_scale * 0.25}, [1]]),
                        dict(label="Arrow: 1x",
                             method="restyle",
                             args=[{"sizeref": unified_arrow_scale * 0.5}, [1]]),
                        dict(label="Arrow: 2x",
                             method="restyle",
                             args=[{"sizeref": unified_arrow_scale * 1.0}, [1]]),
                        dict(label="Arrow: 4x",
                             method="restyle",
                             args=[{"sizeref": unified_arrow_scale * 2.0}, [1]]),
                        dict(label="Arrow: 8x",
                             method="restyle",
                             args=[{"sizeref": unified_arrow_scale * 4.0}, [1]]),
                        dict(label="Arrow: 16x",
                             method="restyle",
                             args=[{"sizeref": unified_arrow_scale * 8.0}, [1]]),
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=1,  # 默认选中 Arrow: 1x
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
                             method="restyle",
                             args=[{"visible": True}, [0, 1]]),
                        dict(label="Points Only",
                             method="update",
                             args=[{"visible": [True, False]}]),
                        dict(label="Arrows Only",
                             method="restyle",
                             args=[{"visible": [False, True]}]),
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    active=0,  # 默认选中 Show Both
                    x=0.38,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1
                ),
                # Play/Pause control
                dict(
                    buttons=[
                        dict(label="▶ Play", method="animate",
                             args=[None, dict(frame=dict(duration=500, redraw=True),
                                            fromcurrent=True,
                                            mode="immediate")]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                              mode="immediate")])
                    ],
                    direction="left",
                    pad=dict(r=10, t=87),
                    showactive=False,
                    type="buttons",
                    x=0.1, xanchor="right", y=0, yanchor="top"
                )
            ]
        )
        
        # Save
        html_path = os.path.join(self.gradient_dir, 'gradient_3d', 
                                'gradient_timeline.html')
        fig.write_html(html_path, include_plotlyjs='cdn')
        
        # 添加自定义JavaScript来保持样式
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        custom_js = f"""
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var gd = document.getElementsByClassName('plotly-graph-div')[0];
            var currentPointSize = 1;
            var currentArrowScale = {unified_arrow_scale * 0.5};
            var isRestoring = false;  // 防止重复恢复
            
            // 监听restyle事件（点云大小、箭头大小调整）
            gd.on('plotly_restyle', function(data) {{
                if (data[0]['marker.size'] !== undefined) {{
                    currentPointSize = data[0]['marker.size'];
                    console.log('Point size changed to:', currentPointSize);
                }}
                if (data[0]['sizeref'] !== undefined) {{
                    currentArrowScale = data[0]['sizeref'];
                    console.log('Arrow scale changed to:', currentArrowScale);
                }}
            }});
            
            // 恢复样式的函数
            function restoreStyles() {{
                if (isRestoring) return;
                isRestoring = true;
                
                setTimeout(function() {{
                    // 恢复点云大小
                    Plotly.restyle(gd, {{'marker.size': currentPointSize}}, [0]);
                    // 恢复箭头大小
                    Plotly.restyle(gd, {{'sizeref': currentArrowScale}}, [1]);
                    console.log('Styles restored - Point:', currentPointSize, 'Arrow:', currentArrowScale);
                    isRestoring = false;
                }}, 50);
            }}
            
            // 监听动画帧变化（Play按钮和Slider都会触发）
            gd.on('plotly_animated', function() {{
                restoreStyles();
            }});
            
            // 监听slider变化（额外保险）
            gd.on('plotly_sliderchange', function() {{
                console.log('Slider changed');
                restoreStyles();
            }});
        }});
        </script>
        """
        
        # 在</body>前插入JavaScript
        html_content = html_content.replace('</body>', custom_js + '\n</body>')
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✓ Timeline visualization saved: {html_path}")
        print(f"  Time points: {len(valid_indices)}")
        print(f"  Points per frame: ~{max_points}")
        print(f"  Interactive controls:")
        print(f"    1. Point Size: Top-left dropdown (0.1x - 5x)")
        print(f"    2. Arrow Size: Second dropdown (0.5x - 16x)")
        print(f"    3. Visibility: Third dropdown (Both/Points/Arrows)")
        print(f"    4. Time Slider: Bottom slider to change time")
        print(f"    5. Play/Pause: Bottom-left buttons")
        print(f"  Visualization:")
        print(f"    • Arrow direction: Gradient direction (normalized)")
        print(f"    • Arrow & Point color: Gradient magnitude (log scale, hot colormap)")
        print(f"    • Arrow length: Uniform (adjustable via dropdown)")
        print(f"  Features:")
        print(f"    ✓ Style persistence during animation (JS-based)")
        print(f"    ✓ Perspective projection with optimized clipping")
        print(f"    ✓ Unified color mapping across all time points")
        print(f"    ✓ Point cloud position changes with deformation at each time")
        print(f"  Debug:")
        print(f"    • Open browser console (F12) to see debug logs")
        print(f"  Open in browser: firefox {html_path}")

