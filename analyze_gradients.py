#!/usr/bin/env python3
"""
梯度分析脚本
用于分析训练后的梯度统计信息
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_gradient_stats(model_path):
    """加载梯度统计信息"""
    stats_path = os.path.join(model_path, 'gradient_vis', 'gradient_stats.json')
    
    if not os.path.exists(stats_path):
        print(f"错误: 未找到梯度统计文件: {stats_path}")
        return None
    
    with open(stats_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def analyze_gradient_trends(gradient_history):
    """分析梯度趋势 - 支持coarse和fine分离的数据"""
    print("\n=== Gradient Trends Analysis ===")
    
    # 检查是否有新格式（coarse/fine分离）
    if 'coarse' in gradient_history and 'fine' in gradient_history:
        # 新格式
        for stage in ['coarse', 'fine']:
            stage_history = gradient_history[stage]
            iterations = stage_history['iterations']
            
            if len(iterations) == 0:
                continue
                
            print(f"\n[{stage.upper()} Stage]")
            print(f"Total records: {len(iterations)}")
            print(f"Iteration range: {min(iterations)} - {max(iterations)}")
            
            # 分析各个梯度类型的趋势
            grad_types = ['xyz', 'opacity', 'scale', 'rotation', 'viewspace', 
                          'deformation_mlp', 'deformation_grid']
            
            for grad_type in grad_types:
                if grad_type in stage_history and len(stage_history[grad_type]) > 0:
                    values = stage_history[grad_type]
                    non_zero_values = [v for v in values if v > 0]
                    
                    if non_zero_values:
                        print(f"\n  {grad_type}:")
                        print(f"    Mean: {np.mean(non_zero_values):.6e}")
                        print(f"    Median: {np.median(non_zero_values):.6e}")
                        print(f"    Max: {np.max(non_zero_values):.6e}")
                        print(f"    Min: {np.min(non_zero_values):.6e}")
                        
                        # 检查趋势
                        if len(non_zero_values) > 10:
                            first_half = np.mean(non_zero_values[:len(non_zero_values)//2])
                            second_half = np.mean(non_zero_values[len(non_zero_values)//2:])
                            change_ratio = (second_half - first_half) / first_half * 100
                            
                            if abs(change_ratio) > 50:
                                trend = "Increase" if change_ratio > 0 else "Decrease"
                                print(f"    Trend: {trend} ({abs(change_ratio):.1f}%)")
    else:
        # 旧格式兼容
        iterations = gradient_history.get('iterations', [])
        if len(iterations) == 0:
            print("No gradient data")
            return
        
        print(f"Total records: {len(iterations)}")
        print(f"Iteration range: {min(iterations)} - {max(iterations)}")


def detect_anomalies(detailed_stats):
    """检测梯度异常"""
    print("\n=== 梯度异常检测 ===")
    
    anomaly_counts = {
        'gradient_vanishing': [],
        'gradient_explosion': [],
        'has_nan': [],
        'has_inf': []
    }
    
    for stats in detailed_stats:
        iteration = stats['iteration']
        
        # 检查主要梯度
        for grad_name, grad_stats in stats['gradients'].items():
            if grad_stats.get('warning'):
                anomaly_counts[grad_stats['warning']].append((iteration, grad_name))
            if grad_stats.get('has_nan'):
                anomaly_counts['has_nan'].append((iteration, grad_name))
            if grad_stats.get('has_inf'):
                anomaly_counts['has_inf'].append((iteration, grad_name))
    
    # 打印异常统计
    total_anomalies = sum(len(v) for v in anomaly_counts.values())
    
    if total_anomalies == 0:
        print("未检测到梯度异常 ✓")
    else:
        print(f"检测到 {total_anomalies} 次异常:")
        
        for anomaly_type, occurrences in anomaly_counts.items():
            if occurrences:
                print(f"\n{anomaly_type}: {len(occurrences)} 次")
                # 显示前5次
                for i, (iter_num, grad_name) in enumerate(occurrences[:5]):
                    print(f"  - Iteration {iter_num}: {grad_name}")
                if len(occurrences) > 5:
                    print(f"  ... 还有 {len(occurrences) - 5} 次")


def plot_gradient_comparison(gradient_history, save_path):
    """绘制梯度对比图 - 支持coarse和fine分离的数据"""
    
    # 检查是否有新格式
    if 'coarse' in gradient_history and 'fine' in gradient_history:
        # 新格式 - 绘制coarse和fine的对比
        has_coarse = len(gradient_history['coarse']['iterations']) > 0
        has_fine = len(gradient_history['fine']['iterations']) > 0
        
        if not has_coarse and not has_fine:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Gradient Analysis (Blue: Coarse, Red: Fine)', fontsize=16)
        
        # 主要参数梯度
        ax1 = axes[0, 0]
        for key in ['xyz', 'opacity', 'scale', 'rotation']:
            if has_coarse and any(v > 0 for v in gradient_history['coarse'][key]):
                ax1.plot(gradient_history['coarse']['iterations'], 
                        gradient_history['coarse'][key], 
                        label=f'{key} (coarse)', linewidth=2, alpha=0.7, color='blue', linestyle='--')
            if has_fine and any(v > 0 for v in gradient_history['fine'][key]):
                ax1.plot(gradient_history['fine']['iterations'], 
                        gradient_history['fine'][key], 
                        label=f'{key} (fine)', linewidth=2, alpha=0.7, color='red')
        ax1.set_yscale('log')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Norm (log scale)')
        ax1.set_title('Gaussian Parameters Gradients')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 特征梯度
        ax2 = axes[0, 1]
        for key in ['viewspace']:
            if has_coarse and any(v > 0 for v in gradient_history['coarse'][key]):
                ax2.plot(gradient_history['coarse']['iterations'], 
                        gradient_history['coarse'][key], 
                        label=f'{key} (coarse)', linewidth=2, alpha=0.7, color='blue', linestyle='--')
            if has_fine and any(v > 0 for v in gradient_history['fine'][key]):
                ax2.plot(gradient_history['fine']['iterations'], 
                        gradient_history['fine'][key], 
                        label=f'{key} (fine)', linewidth=2, alpha=0.7, color='red')
        ax2.set_yscale('log')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.set_title('Viewspace Gradients')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 变形网络梯度
        ax3 = axes[1, 0]
        for key in ['deformation_mlp', 'deformation_grid']:
            if has_coarse and any(v > 0 for v in gradient_history['coarse'][key]):
                ax3.plot(gradient_history['coarse']['iterations'], 
                        gradient_history['coarse'][key], 
                        label=f'{key} (coarse)', linewidth=2, alpha=0.7, color='blue', linestyle='--')
            if has_fine and any(v > 0 for v in gradient_history['fine'][key]):
                ax3.plot(gradient_history['fine']['iterations'], 
                        gradient_history['fine'][key], 
                        label=f'{key} (fine)', linewidth=2, alpha=0.7, color='red')
        ax3.set_yscale('log')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Norm (log scale)')
        ax3.set_title('Deformation Network Gradients')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 所有梯度对比
        ax4 = axes[1, 1]
        all_keys = ['xyz', 'opacity', 'deformation_mlp']
        for key in all_keys:
            if has_coarse and any(v > 0 for v in gradient_history['coarse'][key]):
                ax4.plot(gradient_history['coarse']['iterations'], 
                        gradient_history['coarse'][key], 
                        label=f'{key} (coarse)', linewidth=2, alpha=0.7, color='blue', linestyle='--')
            if has_fine and any(v > 0 for v in gradient_history['fine'][key]):
                ax4.plot(gradient_history['fine']['iterations'], 
                        gradient_history['fine'][key], 
                        label=f'{key} (fine)', linewidth=2, alpha=0.7, color='red')
        ax4.set_yscale('log')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Gradient Norm (log scale)')
        ax4.set_title('All Gradients Comparison')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
    else:
        # 旧格式兼容
        iterations = gradient_history.get('iterations', [])
        if len(iterations) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Gradient Analysis', fontsize=16)
        
        ax1 = axes[0, 0]
        for key in ['xyz', 'opacity', 'scale', 'rotation']:
            if key in gradient_history and any(v > 0 for v in gradient_history[key]):
                ax1.plot(iterations, gradient_history[key], label=key, linewidth=2)
        ax1.set_yscale('log')
        ax1.set_title('Gaussian Parameters')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="分析梯度统计信息")
    parser.add_argument("--model_path", type=str, required=True, help="模型输出路径")
    
    args = parser.parse_args()
    
    print("="*60)
    print("梯度分析工具")
    print("="*60)
    print(f"模型路径: {args.model_path}")
    
    # 加载梯度统计
    data = load_gradient_stats(args.model_path)
    if data is None:
        return 1
    
    gradient_history = data.get('gradient_history', {})
    detailed_stats = data.get('detailed_stats', [])
    
    # 分析梯度趋势
    if gradient_history:
        analyze_gradient_trends(gradient_history)
    
    # 检测异常
    if detailed_stats:
        detect_anomalies(detailed_stats)
    
    # 绘制对比图
    if gradient_history and len(gradient_history.get('iterations', [])) > 0:
        save_path = os.path.join(args.model_path, 'gradient_vis', 'gradient_analysis.png')
        plot_gradient_comparison(gradient_history, save_path)
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

