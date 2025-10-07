#!/usr/bin/env python3
"""
4DGS训练时间可视化分析工具
分析每轮用时曲线和各部分的占比
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys

# 尝试导入seaborn，如果失败则使用matplotlib
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  seaborn未安装，将使用matplotlib绘制热力图")

# 设置字体（使用英文避免字体问题）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TimingVisualizer:
    def __init__(self, timing_report_path):
        """初始化可视化器"""
        self.timing_report_path = Path(timing_report_path)
        self.data = None
        self.load_data()
        
    def load_data(self):
        """加载时间报告数据"""
        try:
            with open(self.timing_report_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ 成功加载时间报告: {self.timing_report_path}")
        except Exception as e:
            print(f"❌ 加载时间报告失败: {e}")
            sys.exit(1)
    
    def extract_iteration_data(self):
        """提取每轮迭代数据"""
        per_iteration = self.data.get('per_iteration_timings', {})
        training_logs = self.data.get('training_logs', [])
        
        iterations = []
        total_times = []
        stages = []
        losses = []
        psnrs = []
        
        # 提取每轮总用时
        for iter_num, timings in per_iteration.items():
            if isinstance(timings, dict):
                for key, time_val in timings.items():
                    if key.endswith('_total'):
                        iterations.append(int(iter_num))
                        total_times.append(time_val)
                        stage = key.replace('_total', '')
                        stages.append(stage)
                        break
        
        # 提取训练指标
        log_dict = {log['iteration']: log for log in training_logs}
        for iter_num in iterations:
            if iter_num in log_dict:
                losses.append(log_dict[iter_num].get('loss', 0))
                psnrs.append(log_dict[iter_num].get('psnr', 0))
            else:
                losses.append(0)
                psnrs.append(0)
        
        return {
            'iterations': iterations,
            'total_times': total_times,
            'stages': stages,
            'losses': losses,
            'psnrs': psnrs
        }
    
    def extract_operation_data(self):
        """提取各操作数据"""
        per_iteration = self.data.get('per_iteration_timings', {})
        
        # 收集所有操作名称
        all_operations = set()
        for timings in per_iteration.values():
            if isinstance(timings, dict):
                for key in timings.keys():
                    if not key.endswith('_total'):
                        all_operations.add(key)
        
        # 按阶段分组操作
        coarse_ops = [op for op in all_operations if op.startswith('coarse_')]
        fine_ops = [op for op in all_operations if op.startswith('fine_')]
        
        # 提取数据
        operation_data = {}
        for op in all_operations:
            operation_data[op] = []
        
        iterations = []
        for iter_num, timings in per_iteration.items():
            if isinstance(timings, dict):
                iterations.append(int(iter_num))
                for op in all_operations:
                    operation_data[op].append(timings.get(op, 0))
        
        return {
            'iterations': sorted(iterations),
            'operations': operation_data,
            'coarse_ops': coarse_ops,
            'fine_ops': fine_ops
        }
    
    def plot_iteration_timing_curve(self, save_path=None):
        """绘制每轮用时曲线"""
        data = self.extract_iteration_data()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 按阶段分组数据
        coarse_data = {'iterations': [], 'times': [], 'losses': [], 'psnrs': []}
        fine_data = {'iterations': [], 'times': [], 'losses': [], 'psnrs': []}
        
        for i, stage in enumerate(data['stages']):
            if stage == 'coarse':
                coarse_data['iterations'].append(data['iterations'][i])
                coarse_data['times'].append(data['total_times'][i])
                coarse_data['losses'].append(data['losses'][i])
                coarse_data['psnrs'].append(data['psnrs'][i])
            elif stage == 'fine':
                fine_data['iterations'].append(data['iterations'][i])
                fine_data['times'].append(data['total_times'][i])
                fine_data['losses'].append(data['losses'][i])
                fine_data['psnrs'].append(data['psnrs'][i])
        
        # 绘制用时曲线
        if coarse_data['iterations']:
            ax1.plot(coarse_data['iterations'], coarse_data['times'], 
                    'o-', label='Coarse阶段', color='blue', alpha=0.7)
        if fine_data['iterations']:
            ax1.plot(fine_data['iterations'], fine_data['times'], 
                    'o-', label='Fine阶段', color='red', alpha=0.7)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Time per Iteration (seconds)')
        ax1.set_title('Training Time per Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制损失和PSNR曲线
        if coarse_data['iterations']:
            ax2_twin = ax2.twinx()
            ax2.plot(coarse_data['iterations'], coarse_data['losses'], 
                    'o-', label='Coarse损失', color='blue', alpha=0.7)
            ax2_twin.plot(coarse_data['iterations'], coarse_data['psnrs'], 
                         's-', label='Coarse PSNR', color='lightblue', alpha=0.7)
        
        if fine_data['iterations']:
            ax2.plot(fine_data['iterations'], fine_data['losses'], 
                    'o-', label='Fine损失', color='red', alpha=0.7)
            ax2_twin.plot(fine_data['iterations'], fine_data['psnrs'], 
                         's-', label='Fine PSNR', color='pink', alpha=0.7)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss', color='black')
        ax2_twin.set_ylabel('PSNR', color='gray')
        ax2.set_title('Training Metrics')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 用时曲线图已保存: {save_path}")
        
        plt.show()
    
    def plot_operation_breakdown(self, save_path=None):
        """绘制操作占比分析"""
        data = self.extract_operation_data()
        
        # 计算平均占比
        avg_percentages = {}
        for op, times in data['operations'].items():
            if times and any(t > 0 for t in times):
                avg_time = np.mean([t for t in times if t > 0])
                avg_percentages[op] = avg_time
        
        # 按阶段分组
        coarse_avg = {op: avg_percentages[op] for op in data['coarse_ops'] if op in avg_percentages}
        fine_avg = {op: avg_percentages[op] for op in data['fine_ops'] if op in avg_percentages}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Coarse阶段占比
        if coarse_avg:
            ops = [op.replace('coarse_', '') for op in coarse_avg.keys()]
            times = list(coarse_avg.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(ops)))
            
            wedges, texts, autotexts = ax1.pie(times, labels=ops, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            ax1.set_title('Coarse Stage Operation Time Distribution')
        
        # Fine阶段占比
        if fine_avg:
            ops = [op.replace('fine_', '') for op in fine_avg.keys()]
            times = list(fine_avg.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(ops)))
            
            wedges, texts, autotexts = ax2.pie(times, labels=ops, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            ax2.set_title('Fine Stage Operation Time Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 操作占比图已保存: {save_path}")
        
        plt.show()
    
    def plot_operation_trends(self, save_path=None):
        """绘制各操作用时趋势"""
        data = self.extract_operation_data()
        
        # 选择主要操作进行可视化
        main_operations = ['data_loading', 'render', 'loss_computation', 'optimizer_step']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, op in enumerate(main_operations):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 绘制coarse和fine阶段的数据
            coarse_op = f'coarse_{op}'
            fine_op = f'fine_{op}'
            
            if coarse_op in data['operations']:
                coarse_times = data['operations'][coarse_op]
                coarse_iters = [iter_num for iter_num, time in zip(data['iterations'], coarse_times) if time > 0]
                coarse_times_filtered = [time for time in coarse_times if time > 0]
                if coarse_iters:
                    ax.plot(coarse_iters, coarse_times_filtered, 'o-', 
                           label='Coarse', color='blue', alpha=0.7)
            
            if fine_op in data['operations']:
                fine_times = data['operations'][fine_op]
                fine_iters = [iter_num for iter_num, time in zip(data['iterations'], fine_times) if time > 0]
                fine_times_filtered = [time for time in fine_times if time > 0]
                if fine_iters:
                    ax.plot(fine_iters, fine_times_filtered, 's-', 
                           label='Fine', color='red', alpha=0.7)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'{op.replace("_", " ").title()} Time Trend')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 操作趋势图已保存: {save_path}")
        
        plt.show()
    
    def plot_heatmap(self, save_path=None):
        """绘制操作用时热力图"""
        data = self.extract_operation_data()
        
        # 准备热力图数据
        operations = []
        iterations = data['iterations']
        
        # 收集所有操作
        for op in data['operations'].keys():
            if not op.endswith('_total'):
                operations.append(op)
        
        # 创建数据矩阵
        matrix = []
        for op in operations:
            times = data['operations'][op]
            matrix.append(times)
        
        if not matrix:
            print("❌ 没有找到操作数据")
            return
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制热力图
        if HAS_SEABORN:
            sns.heatmap(matrix, 
                       xticklabels=iterations,
                       yticklabels=[op.replace('_', ' ').title() for op in operations],
                       cmap='YlOrRd',
                       cbar_kws={'label': '用时 (秒)'},
                       ax=ax)
        else:
            # 使用matplotlib绘制热力图
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(iterations)))
            ax.set_xticklabels(iterations)
            ax.set_yticks(range(len(operations)))
            ax.set_yticklabels([op.replace('_', ' ').title() for op in operations])
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('用时 (秒)')
            
            # 添加数值标注
            for i in range(len(operations)):
                for j in range(len(iterations)):
                    if matrix[i][j] > 0:
                        ax.text(j, i, f'{matrix[i][j]:.2f}', 
                               ha='center', va='center', fontsize=8)
        
        ax.set_title('Operation Time Heatmap')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Operation Type')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 热力图已保存: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self):
        """生成摘要报告"""
        data = self.extract_iteration_data()
        op_data = self.extract_operation_data()
        
        print("\n" + "="*60)
        print("📊 4DGS训练时间分析报告")
        print("="*60)
        
        # 基本统计
        total_training_time = self.data.get('total_training_time', 0)
        print(f"总训练时间: {total_training_time:.2f} 秒 ({total_training_time/60:.1f} 分钟)")
        
        if data['iterations']:
            avg_time = np.mean(data['total_times'])
            min_time = np.min(data['total_times'])
            max_time = np.max(data['total_times'])
            print(f"平均每轮用时: {avg_time:.3f} 秒")
            print(f"最快轮次: {min_time:.3f} 秒")
            print(f"最慢轮次: {max_time:.3f} 秒")
            print(f"总轮次数: {len(data['iterations'])}")
        
        # 各操作平均用时
        print("\n各操作平均用时:")
        for op, times in op_data['operations'].items():
            if times and any(t > 0 for t in times):
                avg_time = np.mean([t for t in times if t > 0])
                print(f"  {op.replace('_', ' ').title()}: {avg_time:.3f} 秒")
        
        # 训练指标
        if data['losses'] and any(l > 0 for l in data['losses']):
            final_loss = data['losses'][-1] if data['losses'][-1] > 0 else data['losses'][-2]
            print(f"\n最终损失: {final_loss:.6f}")
        
        if data['psnrs'] and any(p > 0 for p in data['psnrs']):
            final_psnr = data['psnrs'][-1] if data['psnrs'][-1] > 0 else data['psnrs'][-2]
            print(f"最终PSNR: {final_psnr:.2f}")
        
        print("="*60)
    
    def run_full_analysis(self, output_dir=None):
        """运行完整分析"""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        print("🚀 开始4DGS训练时间可视化分析...")
        
        # 生成摘要报告
        self.generate_summary_report()
        
        # 生成各种图表
        self.plot_iteration_timing_curve(
            save_path=output_dir / "iteration_timing_curve.png" if output_dir else None
        )
        
        self.plot_operation_breakdown(
            save_path=output_dir / "operation_breakdown.png" if output_dir else None
        )
        
        self.plot_operation_trends(
            save_path=output_dir / "operation_trends.png" if output_dir else None
        )
        
        self.plot_heatmap(
            save_path=output_dir / "operation_heatmap.png" if output_dir else None
        )
        
        print("✅ 分析完成！")

def main():
    parser = argparse.ArgumentParser(description='4DGS训练时间可视化分析工具')
    parser.add_argument('timing_report', help='时间报告JSON文件路径')
    parser.add_argument('--output', '-o', help='输出目录（可选）')
    parser.add_argument('--curve', action='store_true', help='只显示用时曲线')
    parser.add_argument('--breakdown', action='store_true', help='只显示操作占比')
    parser.add_argument('--trends', action='store_true', help='只显示操作趋势')
    parser.add_argument('--heatmap', action='store_true', help='只显示热力图')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = TimingVisualizer(args.timing_report)
    
    # 根据参数选择显示内容
    if args.curve:
        visualizer.plot_iteration_timing_curve()
    elif args.breakdown:
        visualizer.plot_operation_breakdown()
    elif args.trends:
        visualizer.plot_operation_trends()
    elif args.heatmap:
        visualizer.plot_heatmap()
    else:
        # 显示所有图表
        visualizer.run_full_analysis(args.output)

if __name__ == "__main__":
    main()
