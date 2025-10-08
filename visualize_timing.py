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

# 设置字体（尝试中文字体，失败则使用英文）
try:
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
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
        """提取每轮迭代数据（调整fine阶段的iteration编号，使其连续）"""
        per_iteration = self.data.get('per_iteration_timings', {})
        training_logs = self.data.get('training_logs', [])
        
        # 简化逻辑：根据training_logs判断stage范围
        coarse_iters_set = set()
        fine_iters_set = set()
        
        for log in training_logs:
            iter_num = log.get('iteration')
            stage = log.get('stage')
            if iter_num and stage:
                if stage == 'coarse':
                    coarse_iters_set.add(iter_num)
                elif stage == 'fine':
                    fine_iters_set.add(iter_num)
        
        # Coarse阶段的最大iteration
        coarse_max_iter = max(coarse_iters_set) if coarse_iters_set else 3000
        fine_max_iter = max(fine_iters_set) if fine_iters_set else 14000
        
        iterations = []
        total_times = []
        stages = []
        losses = []
        psnrs = []
        actual_iterations = []  # 实际的iteration编号（用于查找log）
        
        # 遍历per_iteration_timings，提取数据
        for iter_num_str in sorted(per_iteration.keys(), key=lambda x: int(x)):
            timings = per_iteration[iter_num_str]
            if isinstance(timings, dict):
                actual_iter = int(iter_num_str)
                
                # 简单判断：
                # Coarse: iteration 1-3000，使用coarse_total
                # Fine: iteration 1-14000，使用fine_total，调整为3001-17000
                if actual_iter <= coarse_max_iter:
                    # Coarse阶段
                    if 'coarse_total' in timings and timings['coarse_total'] > 0:
                        iterations.append(actual_iter)
                        actual_iterations.append(actual_iter)
                        total_times.append(timings['coarse_total'])
                        stages.append('coarse')
                        
                if actual_iter <= fine_max_iter:
                    # Fine阶段（iteration 1-14000映射到3001-17000）
                    if 'fine_total' in timings and timings['fine_total'] > 0:
                        adjusted_iter = actual_iter + coarse_max_iter
                        iterations.append(adjusted_iter)
                        actual_iterations.append(actual_iter)
                        total_times.append(timings['fine_total'])
                        stages.append('fine')
        
        # 提取训练指标（只提取有记录的iteration，避免填充0值）
        # training_logs中只记录了每10步，其他iteration不应该显示
        for i, actual_iter in enumerate(actual_iterations):
            stage = stages[i]
            # 在training_logs中找到对应的记录
            found = False
            for log in training_logs:
                if log.get('iteration') == actual_iter and log.get('stage') == stage:
                    losses.append(log.get('loss', None))
                    psnrs.append(log.get('psnr', None))
                    found = True
                    break
            if not found:
                # 使用None而不是0，这样绘图时会自动跳过
                losses.append(None)
                psnrs.append(None)
        
        return {
            'iterations': iterations,
            'total_times': total_times,
            'stages': stages,
            'losses': losses,
            'psnrs': psnrs
        }
    
    def extract_operation_data(self):
        """提取各操作数据（调整fine阶段的iteration编号，使其连续）"""
        per_iteration = self.data.get('per_iteration_timings', {})
        training_logs = self.data.get('training_logs', [])
        
        # 找到coarse和fine阶段的最大iteration
        coarse_iters_in_log = set()
        fine_iters_in_log = set()
        
        for log in training_logs:
            iter_num = log.get('iteration')
            stage = log.get('stage')
            if iter_num and stage:
                if stage == 'coarse':
                    coarse_iters_in_log.add(iter_num)
                elif stage == 'fine':
                    fine_iters_in_log.add(iter_num)
        
        coarse_max_iter = max(coarse_iters_in_log) if coarse_iters_in_log else 3000
        fine_max_iter = max(fine_iters_in_log) if fine_iters_in_log else 14000
        
        # 收集所有操作名称，排除iteration（因为它是总时间，会导致重复计算）
        all_operations = set()
        for timings in per_iteration.values():
            if isinstance(timings, dict):
                for key in timings.keys():
                    if not key.endswith('_total') and not key.endswith('_iteration'):
                        all_operations.add(key)
        
        # 按阶段分组操作
        coarse_ops = [op for op in all_operations if op.startswith('coarse_')]
        fine_ops = [op for op in all_operations if op.startswith('fine_')]
        
        # 提取数据（和extract_iteration_data逻辑一致）
        # 关键：每个timings同时包含coarse和fine的所有操作，需要分别提取
        
        iterations = []
        operation_data = {}
        for op in all_operations:
            operation_data[op] = []
        
        for iter_num_str in sorted(per_iteration.keys(), key=lambda x: int(x)):
            timings = per_iteration[iter_num_str]
            if isinstance(timings, dict):
                actual_iter = int(iter_num_str)
                
                # Coarse: iteration 1-3000
                if actual_iter <= coarse_max_iter:
                    iterations.append(actual_iter)
                    # 只提取coarse操作的数据
                    for op in all_operations:
                        operation_data[op].append(timings.get(op, 0))
                
                # Fine: iteration 1-14000，调整为3001-17000
                if actual_iter <= fine_max_iter:
                    adjusted_iter = actual_iter + coarse_max_iter
                    iterations.append(adjusted_iter)
                    # 只提取fine操作的数据（注意：coarse操作会是0）
                for op in all_operations:
                    operation_data[op].append(timings.get(op, 0))
        
        return {
            'iterations': iterations,
            'operations': operation_data,
            'coarse_ops': coarse_ops,
            'fine_ops': fine_ops,
            'coarse_max_iter': coarse_max_iter  # 返回coarse最大iteration，供绘图使用
        }
    
    def plot_iteration_timing_curve(self, save_path=None, save_path_full=None):
        """绘制每轮用时曲线（生成两个版本：98分位数版本 + 断轴完整版本）"""
        data = self.extract_iteration_data()
        
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
        
        # ========== 版本1：98分位数视图 ==========
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 绘制用时曲线 - 处理异常值，使用散点图
        if coarse_data['iterations']:
            ax1.scatter(coarse_data['iterations'], coarse_data['times'], 
                       label='Coarse Stage', color='blue', alpha=0.6, s=20, edgecolors='none')
            
        if fine_data['iterations']:
            ax1.scatter(fine_data['iterations'], fine_data['times'], 
                       label='Fine Stage', color='red', alpha=0.6, s=20, edgecolors='none')
        
        # 设置合理的y轴上限和下限（避免极端值压缩数据，减少空白区域）
        all_times = coarse_data['times'] + fine_data['times']
        if all_times:
            y_min = np.percentile(all_times, 5) * 0.9  # 使用5分位数的90%作为下限
            y_max = np.percentile(all_times, 98)  # 使用98分位数作为上限
            outliers_count = np.sum(np.array(all_times) > y_max)
            max_val = np.max(all_times)
            ax1.set_ylim(y_min, y_max * 1.1)
            
            # # 添加离群点提示
            # if outliers_count > 0:
            #     ax1.text(0.98, 0.98, f'⚠️ {outliers_count} outliers\n(max: {max_val:.3f}s)', 
            #            transform=ax1.transAxes, fontsize=9, 
            #            verticalalignment='top', horizontalalignment='right',
            #            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax1.set_xlabel('Iteration (Coarse: 1-3000, Fine: 3001-17000)', fontsize=11)
        ax1.set_ylabel('Time per Iteration (seconds)', fontsize=11)
        ax1.set_title('Training Time per Iteration ', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 绘制损失和PSNR曲线 - 使用折线图（过滤None值）
        ax2_twin = ax2.twinx()
        
        if coarse_data['iterations']:
            # 过滤None值
            valid_coarse = [(it, loss, psnr) for it, loss, psnr in 
                           zip(coarse_data['iterations'], coarse_data['losses'], coarse_data['psnrs'])
                           if loss is not None and psnr is not None]
            if valid_coarse:
                coarse_its, coarse_losses, coarse_psnrs = zip(*valid_coarse)
                ax2.plot(coarse_its, coarse_losses, 
                        label='Coarse Loss', color='blue', alpha=0.8, linewidth=2)
                ax2_twin.plot(coarse_its, coarse_psnrs, 
                             label='Coarse PSNR', color='lightblue', alpha=0.8, 
                             linewidth=2)
        
        if fine_data['iterations']:
            # 过滤None值
            valid_fine = [(it, loss, psnr) for it, loss, psnr in 
                         zip(fine_data['iterations'], fine_data['losses'], fine_data['psnrs'])
                         if loss is not None and psnr is not None]
            if valid_fine:
                fine_its, fine_losses, fine_psnrs = zip(*valid_fine)
                ax2.plot(fine_its, fine_losses, 
                        label='Fine Loss', color='red', alpha=0.8, linewidth=2)
                ax2_twin.plot(fine_its, fine_psnrs, 
                             label='Fine PSNR', color='pink', alpha=0.8, 
                             linewidth=2)
        
        # 优化loss的y轴范围，使对比更明显（去掉最开始的高loss值）
        all_losses = [l for l in coarse_data['losses'] + fine_data['losses'] if l is not None and l > 0]
        if all_losses:
            loss_min = min(all_losses)
            loss_max = max(all_losses)
            # 使用10分位数作为下限，去掉最开始的极高值，突出后期变化
            loss_p10 = np.percentile(all_losses, 10)
            loss_p95 = np.percentile(all_losses, 95)
            # 设置紧凑的范围
            ax2.set_ylim(0, loss_p95 * 1.2)
        
        ax2.set_xlabel('Iteration (Coarse: 1-3000, Fine: 3001-17000)', fontsize=11)
        ax2.set_ylabel('Loss', color='black', fontsize=11)
        ax2_twin.set_ylabel('PSNR (dB)', color='gray', fontsize=11)
        ax2.set_title('Training Metrics (Loss & PSNR)', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper left', fontsize=10)
        ax2_twin.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 用时曲线图（98分位数）已保存: {save_path}")
        plt.show()
        plt.close()
        
        # ========== 版本2：断轴完整视图（显示所有极端值） ==========
        if save_path_full and all_times:
            self._plot_iteration_timing_curve_broken_axis(coarse_data, fine_data, all_times, save_path_full)
    
    def _plot_iteration_timing_curve_broken_axis(self, coarse_data, fine_data, all_times, save_path):
        """绘制带断轴的完整用时曲线（显示所有极端值）"""
        from matplotlib.gridspec import GridSpec
        
        all_times_array = np.array(all_times)
        p90 = np.percentile(all_times_array, 90)
        p98 = np.percentile(all_times_array, 98)
        max_val = np.max(all_times_array)
        
        # 如果最大值远大于90分位数，使用断轴
        if max_val > p90 * 1.5:
            fig = plt.figure(figsize=(14, 12))
            
            # 手动设置子图位置，按1:3:2比例，断轴空隙极小
            gap_break = 0.005  # 上中图之间极小（断轴空隙）
            gap_metrics = 0.10  # 中下图之间较大（训练指标空隙）- 增大
            
            # 统一的左右边距（下图有两个y轴，需要更多右边距）
            left_margin = 0.10
            right_margin = 0.12  # 为twin axis留出空间
            plot_width = 1.0 - left_margin - right_margin  # 0.78
            
            # 计算高度（总可用空间 0.05-0.93 = 0.88，减去空隙后按1:3:2分配）
            total_height = 0.88 - gap_break - gap_metrics  # 0.765
            h_top = total_height * 1/6  # 约0.127
            h_middle = total_height * 3/6  # 约0.382
            h_bottom = total_height * 2/6  # 约0.255
            
            # 三个图都使用相同的left和width，确保左右边界对齐
            ax_top = fig.add_subplot(3, 1, 1)
            ax_top.set_position([left_margin, 0.93-h_top, plot_width, h_top])
            
            ax_bottom = fig.add_subplot(3, 1, 2)
            ax_bottom.set_position([left_margin, 0.93-h_top-gap_break-h_middle, plot_width, h_middle])
            
            ax_metrics = fig.add_subplot(3, 1, 3)
            ax_metrics.set_position([left_margin, 0.05, plot_width, h_bottom])
            
            # 设置y轴范围
            y_bottom_min = np.percentile(all_times_array, 5) * 0.9  # 使用5分位数的90%作为下限，减少空白
            y_bottom_max = p90 * 1.2
            y_top_min = p98 * 0.95
            
            # 在两个子图中绘制相同的数据
            for ax in [ax_top, ax_bottom]:
                if coarse_data['iterations']:
                    ax.scatter(coarse_data['iterations'], coarse_data['times'], 
                              label='Coarse' if ax == ax_bottom else None,
                              color='blue', alpha=0.6, s=15, edgecolors='none')
                if fine_data['iterations']:
                    ax.scatter(fine_data['iterations'], fine_data['times'], 
                              label='Fine' if ax == ax_bottom else None,
                              color='red', alpha=0.6, s=15, edgecolors='none')
            
            # 设置y轴范围
            ax_bottom.set_ylim(y_bottom_min, y_bottom_max)
            ax_top.set_ylim(y_top_min, max_val * 1.05)
            
            # 隐藏上图x轴
            ax_top.set_xticklabels([])
            ax_top.tick_params(labelbottom=False, bottom=False)
            
            # 添加断轴标记
            d = 0.015
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.5)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
            
            kwargs.update(transform=ax_bottom.transAxes)
            ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            
            # 设置标签
            ax_bottom.set_xlabel('Iteration (Coarse: 1-3000, Fine: 3001-17000)', fontsize=11)
            ax_bottom.set_ylabel('Time (seconds)', fontsize=10)
            ax_top.set_ylabel('Time (sec)', fontsize=10)
            ax_top.set_title('Training Time per Iteration (Full View with Broken Axis)', 
                           fontsize=13, fontweight='bold', pad=15)
            ax_bottom.legend(fontsize=10, loc='upper left')
            ax_bottom.grid(True, alpha=0.3)
            ax_top.grid(True, alpha=0.3)
            
            # 绘制训练指标 - 使用折线图（过滤None值）
            ax_metrics_twin = ax_metrics.twinx()
            
            if coarse_data['iterations']:
                valid_coarse_metrics = [(it, loss, psnr) for it, loss, psnr in 
                                       zip(coarse_data['iterations'], coarse_data['losses'], coarse_data['psnrs'])
                                       if loss is not None and psnr is not None]
                if valid_coarse_metrics:
                    c_its, c_losses, c_psnrs = zip(*valid_coarse_metrics)
                    ax_metrics.plot(c_its, c_losses, 
                                   label='Coarse Loss', color='blue', alpha=0.8, linewidth=2)
                    ax_metrics_twin.plot(c_its, c_psnrs, 
                                        label='Coarse PSNR', color='lightblue', alpha=0.8, 
                                        linewidth=2)
            
            if fine_data['iterations']:
                valid_fine_metrics = [(it, loss, psnr) for it, loss, psnr in 
                                     zip(fine_data['iterations'], fine_data['losses'], fine_data['psnrs'])
                                     if loss is not None and psnr is not None]
                if valid_fine_metrics:
                    f_its, f_losses, f_psnrs = zip(*valid_fine_metrics)
                    ax_metrics.plot(f_its, f_losses, 
                                   label='Fine Loss', color='red', alpha=0.8, linewidth=2)
                    ax_metrics_twin.plot(f_its, f_psnrs, 
                                        label='Fine PSNR', color='pink', alpha=0.8, 
                                        linewidth=2)
            
            # 优化loss的y轴范围，使对比更明显（去掉最开始的高loss值）
            all_losses_full = [l for l in coarse_data['losses'] + fine_data['losses'] if l is not None and l > 0]
            if all_losses_full:
                loss_p95_full = np.percentile(all_losses_full, 95)
                # 从0开始，到95分位数，自动裁剪初期的极高loss值
                ax_metrics.set_ylim(0, loss_p95_full * 1.2)
            
            ax_metrics.set_xlabel('Iteration', fontsize=11)
            ax_metrics.set_ylabel('Loss', fontsize=10)
            ax_metrics_twin.set_ylabel('PSNR (dB)', fontsize=10)
            ax_metrics.set_title('Training Metrics (Loss & PSNR)', fontsize=13, fontweight='bold')
            ax_metrics.legend(loc='upper left', fontsize=9)
            ax_metrics_twin.legend(loc='upper right', fontsize=9)
            ax_metrics.grid(True, alpha=0.3)
            
            # 不使用tight_layout，因为我们已经手动设置了position
            # plt.tight_layout()  # 会覆盖手动设置的position
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 用时曲线图（断轴完整版）已保存: {save_path}")
            plt.show()
            plt.close()
        else:
            print("⚠️  数据没有明显极端值，无需断轴显示")
    
    def plot_stage_time_comparison(self, save_path=None):
        """绘制Coarse vs Fine阶段总时间占比"""
        data = self.extract_iteration_data()
        
        # 计算各阶段总时间
        coarse_total_time = 0
        fine_total_time = 0
        
        for i, stage in enumerate(data['stages']):
            if stage == 'coarse':
                coarse_total_time += data['total_times'][i]
            elif stage == 'fine':
                fine_total_time += data['total_times'][i]
        
        total_time = coarse_total_time + fine_total_time
        
        # 绘制饼图
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        stages = ['Coarse Stage', 'Fine Stage']
        times = [coarse_total_time, fine_total_time]
        percentages = [t/total_time*100 for t in times]
        colors = ['#3498db', '#e74c3c']
        
        wedges, texts, autotexts = ax.pie(times, labels=stages, autopct='%1.1f%%',
                                          colors=colors, startangle=90,
                                          textprops={'fontsize': 13, 'fontweight': 'bold'},
                                          pctdistance=0.75, labeldistance=1.1)
        
        # 设置标题和图例
        ax.set_title('Training Time: Coarse vs Fine Stage Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 添加详细信息文本框
        info_text = f'Coarse: {coarse_total_time:.1f}s ({percentages[0]:.1f}%)\n'
        info_text += f'Fine: {fine_total_time:.1f}s ({percentages[1]:.1f}%)\n'
        info_text += f'Total: {total_time:.1f}s'
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 阶段时间对比图已保存: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_operation_breakdown(self, save_path=None):
        """绘制操作占比分析（饼图中标注序号，图例显示序号对应的操作名称）"""
        data = self.extract_operation_data()
        
        # 计算平均占比
        avg_percentages = {}
        for op, times in data['operations'].items():
            if times and any(t > 0 for t in times):
                avg_time = np.mean([t for t in times if t > 0])
                avg_percentages[op] = avg_time
        
        # 按阶段分组并按时间排序
        coarse_avg = {op: avg_percentages[op] for op in data['coarse_ops'] if op in avg_percentages}
        fine_avg = {op: avg_percentages[op] for op in data['fine_ops'] if op in avg_percentages}
        
        # 排序（从大到小）
        coarse_avg = dict(sorted(coarse_avg.items(), key=lambda x: x[1], reverse=True))
        fine_avg = dict(sorted(fine_avg.items(), key=lambda x: x[1], reverse=True))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Coarse阶段占比
        if coarse_avg:
            ops = [op.replace('coarse_', '').replace('_', ' ').title() for op in coarse_avg.keys()]
            times = list(coarse_avg.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(ops)))
            
            # 计算百分比，只在>=3%的扇区显示序号标签，避免重叠
            total_time = sum(times)
            percentages = [t/total_time*100 for t in times]
            labels = [str(i+1) if percentages[i] >= 3 else '' for i in range(len(ops))]
            
            # 自定义百分比显示函数
            def make_autopct(values):
                def my_autopct(pct):
                    return f'{pct:.1f}%' if pct >= 3 else ''
                return my_autopct
            
            wedges, texts, autotexts = ax1.pie(times, labels=labels, autopct=make_autopct(percentages), 
                                             colors=colors, startangle=90, 
                                             textprops={'fontsize': 10, 'fontweight': 'bold'},
                                             pctdistance=0.75, labeldistance=1.08)
            
            # 图例显示序号对应的操作名称（按占比从大到小）
            legend_labels = [f'{i+1}. {op} ({percentages[i]:.1f}%)' for i, op in enumerate(ops)]
            ax1.legend(legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                      fontsize=9, frameon=True, shadow=True)
            ax1.set_title('Coarse Stage Operation Time Distribution', fontsize=13, fontweight='bold', pad=20)
        
        # Fine阶段占比
        if fine_avg:
            ops = [op.replace('fine_', '').replace('_', ' ').title() for op in fine_avg.keys()]
            times = list(fine_avg.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(ops)))
            
            # 计算百分比，只在>=3%的扇区显示序号标签，避免重叠
            total_time = sum(times)
            percentages = [t/total_time*100 for t in times]
            labels = [str(i+1) if percentages[i] >= 3 else '' for i in range(len(ops))]
            
            def make_autopct(values):
                def my_autopct(pct):
                    return f'{pct:.1f}%' if pct >= 3 else ''
                return my_autopct
            
            wedges, texts, autotexts = ax2.pie(times, labels=labels, autopct=make_autopct(percentages), 
                                             colors=colors, startangle=90,
                                             textprops={'fontsize': 10, 'fontweight': 'bold'},
                                             pctdistance=0.75, labeldistance=1.08)
            
            legend_labels = [f'{i+1}. {op} ({percentages[i]:.1f}%)' for i, op in enumerate(ops)]
            ax2.legend(legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                      fontsize=9, frameon=True, shadow=True)
            ax2.set_title('Fine Stage Operation Time Distribution', fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 操作占比图已保存: {save_path}")
        
        plt.show()
    
    def plot_operation_trends(self, save_path=None):
        """绘制各操作用时趋势（横轴连续：Coarse在前，Fine在后）"""
        data = self.extract_operation_data()
        coarse_max_iter = data.get('coarse_max_iter', 3000)
        
        # 选择主要操作进行可视化
        main_operations = ['data_loading', 'render', 'loss_computation', 'optimizer_step']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        for i, op in enumerate(main_operations):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 收集coarse和fine的数据（根据调整后的iteration分离）
            coarse_op = f'coarse_{op}'
            fine_op = f'fine_{op}'
            
            all_times = []
            coarse_data = {'iters': [], 'times': []}
            fine_data = {'iters': [], 'times': []}
            
            # 处理coarse数据（iteration <= coarse_max_iter）
            if coarse_op in data['operations']:
                coarse_times = data['operations'][coarse_op]
                for iter_num, time in zip(data['iterations'], coarse_times):
                    if time > 0 and iter_num <= coarse_max_iter:
                        coarse_data['iters'].append(iter_num)
                        coarse_data['times'].append(time)
                        all_times.append(time)
            
            # 处理fine数据（iteration > coarse_max_iter）
            if fine_op in data['operations']:
                fine_times = data['operations'][fine_op]
                for iter_num, time in zip(data['iterations'], fine_times):
                    if time > 0 and iter_num > coarse_max_iter:
                        fine_data['iters'].append(iter_num)
                        fine_data['times'].append(time)
                        all_times.append(time)
            
            if not all_times:
                continue
            
            # 计算统计信息
            all_times_array = np.array(all_times)
            p95 = np.percentile(all_times_array, 95)
            max_val = np.max(all_times_array)
            median_val = np.median(all_times_array)
            
            # 绘制数据（使用散点图）
            if coarse_data['iters']:
                ax.scatter(coarse_data['iters'], coarse_data['times'], 
                          label='Coarse', color='blue', alpha=0.6, s=15, edgecolors='none')
            
            if fine_data['iters']:
                ax.scatter(fine_data['iters'], fine_data['times'], 
                          label='Fine', color='red', alpha=0.6, s=15, marker='s', edgecolors='none')
            
            # 设置y轴上限为98分位数，避免极端值压缩数据
            y_limit = np.percentile(all_times_array, 98) * 1.15
            ax.set_ylim(0, y_limit)
            
            # 如果有超出范围的极端值，添加标注
            outliers_count = np.sum(all_times_array > y_limit)
            if outliers_count > 0:
                ax.text(0.98, 0.98, f'⚠️ {outliers_count} outliers\n(max: {max_val:.3f}s)', 
                       transform=ax.transAxes, fontsize=8, 
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 添加中位数参考线
            ax.axhline(y=median_val, color='green', linestyle='--', alpha=0.3, linewidth=1)
            ax.text(0.02, median_val, f'median: {median_val:.3f}s', 
                   fontsize=7, color='green', verticalalignment='bottom')
            
            ax.set_xlabel(f'Iteration (Coarse: 1-{coarse_max_iter}, Fine: {coarse_max_iter+1}-17000)', fontsize=10)
            ax.set_ylabel('Time (seconds)', fontsize=10)
            ax.set_title(f'{op.replace("_", " ").title()} Time Trend', 
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 操作趋势图已保存: {save_path}")
        
        plt.show()
    
    def plot_heatmap(self, save_path=None):
        """绘制操作用时热力图（优化性能：采样数据点，不显示文字标注）"""
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
        
        matrix = np.array(matrix)
        
        # 性能优化：如果iteration数量过多（>100），进行采样
        max_iterations_display = 100
        if len(iterations) > max_iterations_display:
            # 均匀采样
            sample_indices = np.linspace(0, len(iterations)-1, max_iterations_display, dtype=int)
            iterations_sampled = [iterations[i] for i in sample_indices]
            matrix = matrix[:, sample_indices]
            print(f"⚡ 热力图采样优化：从 {len(iterations)} 个iterations采样到 {max_iterations_display} 个")
        else:
            iterations_sampled = iterations
        
        # 创建热力图
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 绘制热力图（不添加文字标注以提升性能）
        if HAS_SEABORN:
            sns.heatmap(matrix, 
                       xticklabels=[str(it) for it in iterations_sampled][::max(1, len(iterations_sampled)//20)],
                       yticklabels=[op.replace('_', ' ').title() for op in operations],
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Time (seconds)'},
                       ax=ax,
                       annot=False)  # 不显示数值，提升性能
        else:
            # 使用matplotlib绘制热力图
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            
            # 只显示部分x轴标签，避免拥挤
            tick_step = max(1, len(iterations_sampled) // 20)
            ax.set_xticks(range(0, len(iterations_sampled), tick_step))
            ax.set_xticklabels([str(iterations_sampled[i]) for i in range(0, len(iterations_sampled), tick_step)], 
                              fontsize=8, rotation=45)
            
            ax.set_yticks(range(len(operations)))
            ax.set_yticklabels([op.replace('_', ' ').title() for op in operations], fontsize=9)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Time (seconds)', fontsize=10)
            
            # 不添加数值标注，提升性能
        
        ax.set_title('Operation Time Heatmap (Hover for details)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Iteration (sampled for performance)', fontsize=11)
        ax.set_ylabel('Operation Type', fontsize=11)
        
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
        
        # 训练指标（过滤None值）
        valid_losses = [l for l in data['losses'] if l is not None and l > 0]
        if valid_losses:
            final_loss = valid_losses[-1]
            print(f"\n最终损失: {final_loss:.6f}")
        
        valid_psnrs = [p for p in data['psnrs'] if p is not None and p > 0]
        if valid_psnrs:
            final_psnr = valid_psnrs[-1]
            print(f"最终PSNR: {final_psnr:.2f}")
        
        print("="*60)
    
    def run_full_analysis(self, output_dir=None):
        """运行完整分析（带性能监控，V3.0版本）"""
        import time
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        print("🚀 开始4DGS训练时间可视化分析 (V3.0)...")
        
        # 生成摘要报告
        self.generate_summary_report()
        
        # 生成各种图表并计时
        chart_times = {}
        
        print("\n生成图表中...")
        
        # Coarse vs Fine阶段时间对比
        start = time.time()
        self.plot_stage_time_comparison(
            save_path=output_dir / "stage_time_comparison.png" if output_dir else None
        )
        chart_times['阶段时间对比'] = time.time() - start
        print(f"  ⏱️  阶段时间对比: {chart_times['阶段时间对比']:.2f}秒")
        
        # 迭代时间曲线（两个版本）
        start = time.time()
        self.plot_iteration_timing_curve(
            save_path=output_dir / "iteration_timing_curve.png" if output_dir else None,
            save_path_full=output_dir / "iteration_timing_curve_full.png" if output_dir else None
        )
        chart_times['迭代时间曲线（两个版本）'] = time.time() - start
        print(f"  ⏱️  迭代时间曲线（两个版本）: {chart_times['迭代时间曲线（两个版本）']:.2f}秒")
        
        # 操作占比图
        start = time.time()
        self.plot_operation_breakdown(
            save_path=output_dir / "operation_breakdown.png" if output_dir else None
        )
        chart_times['操作占比图'] = time.time() - start
        print(f"  ⏱️  操作占比图: {chart_times['操作占比图']:.2f}秒")
        
        # 操作趋势图
        start = time.time()
        self.plot_operation_trends(
            save_path=output_dir / "operation_trends.png" if output_dir else None
        )
        chart_times['操作趋势图'] = time.time() - start
        print(f"  ⏱️  操作趋势图: {chart_times['操作趋势图']:.2f}秒")
        
        # 去掉热力图（用户反馈没什么用）
        # self.plot_heatmap(...) - 已移除
        
        total_time = sum(chart_times.values())
        print(f"\n✅ 分析完成！总耗时: {total_time:.2f}秒")
        print(f"📊 生成图表：")
        print(f"  - stage_time_comparison.png (Coarse vs Fine时间占比)")
        print(f"  - iteration_timing_curve.png (98分位数视图)")
        print(f"  - iteration_timing_curve_full.png (断轴完整视图)")
        print(f"  - operation_breakdown.png (操作时间占比)")
        print(f"  - operation_trends.png (操作趋势，横轴已修正)")
        print(f"\n📝 改进说明：")
        print(f"  ✅ 横轴连续性：Fine阶段接在Coarse后（3000+）")
        print(f"  ✅ 断轴视图：间距已增加，横坐标不再重叠")
        print(f"  ✅ 新增阶段对比饼图")

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
