#!/usr/bin/env python3
"""
Debug结果分析工具
分析网格剪枝的效果和性能提升
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

def parse_log_file(log_path):
    """解析训练日志文件"""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    info = {
        'log_path': log_path,
        'initial_points': None,
        'final_points': None,
        'pruning_info': {},
        'training_metrics': [],
        'psnr_values': [],
        'loss_values': []
    }
    
    # 提取初始点云数量
    match = re.search(r'Number of points at initialisation\s*:\s*(\d+)', content)
    if match:
        info['initial_points'] = int(match.group(1))
    
    # 提取网格剪枝信息
    pruning_matches = re.findall(r'\[Grid Pruning\]\s*(.+)', content)
    if pruning_matches:
        for line in pruning_matches:
            if '原始点云数量' in line:
                match = re.search(r'(\d+)', line)
                if match:
                    info['pruning_info']['original_points'] = int(match.group(1))
            elif '剪枝后点云数量' in line:
                match = re.search(r'(\d+)', line)
                if match:
                    info['pruning_info']['pruned_points'] = int(match.group(1))
            elif '点云减少比例' in line:
                match = re.search(r'([\d.]+)%', line)
                if match:
                    info['pruning_info']['reduction_ratio'] = float(match.group(1))
            elif '体素大小' in line:
                match = re.search(r'([\d.]+)', line)
                if match:
                    info['pruning_info']['voxel_size'] = float(match.group(1))
    
    # 提取PSNR值
    psnr_matches = re.findall(r'PSNR\s*[:\s]+([\d.]+)', content)
    info['psnr_values'] = [float(p) for p in psnr_matches]
    
    # 提取Loss值
    loss_matches = re.findall(r'Loss\s*[:\s]+([\d.]+)', content)
    info['loss_values'] = [float(l) for l in loss_matches]
    
    return info

def analyze_checkpoint(checkpoint_dir):
    """分析模型检查点"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    info = {
        'checkpoint_dir': checkpoint_dir,
        'ply_file': None,
        'model_size': 0,
        'file_count': 0
    }
    
    # 查找.ply文件
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.ply'):
            ply_path = os.path.join(checkpoint_dir, file)
            info['ply_file'] = ply_path
            info['model_size'] = os.path.getsize(ply_path) / (1024*1024)  # MB
    
    # 统计文件数量
    info['file_count'] = len(os.listdir(checkpoint_dir))
    
    return info

def compare_results(baseline_info, pruning_info):
    """对比分析结果"""
    comparison = {
        'point_reduction': None,
        'model_size_reduction': None,
        'psnr_diff': None,
        'loss_diff': None
    }
    
    # 点云数量对比
    if baseline_info and baseline_info['initial_points'] and \
       pruning_info and pruning_info['initial_points']:
        baseline_pts = baseline_info['initial_points']
        pruning_pts = pruning_info['initial_points']
        comparison['point_reduction'] = (1 - pruning_pts / baseline_pts) * 100
    
    # PSNR对比（取最后几个值的平均）
    if baseline_info and baseline_info['psnr_values'] and \
       pruning_info and pruning_info['psnr_values']:
        baseline_psnr = sum(baseline_info['psnr_values'][-5:]) / min(5, len(baseline_info['psnr_values']))
        pruning_psnr = sum(pruning_info['psnr_values'][-5:]) / min(5, len(pruning_info['psnr_values']))
        comparison['psnr_diff'] = pruning_psnr - baseline_psnr
    
    return comparison

def generate_report(baseline_info, pruning_info, baseline_ckpt, pruning_ckpt, output_path):
    """生成详细分析报告"""
    
    report = []
    report.append("=" * 80)
    report.append("网格剪枝 Debug 测试 - 详细分析报告")
    report.append("=" * 80)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Baseline结果
    report.append("-" * 80)
    report.append("测试1: 不使用网格剪枝 (Baseline)")
    report.append("-" * 80)
    
    if baseline_info:
        report.append(f"日志文件: {baseline_info['log_path']}")
        report.append(f"初始点云数量: {baseline_info['initial_points']:,}" if baseline_info['initial_points'] else "初始点云数量: 未知")
        
        if baseline_info['psnr_values']:
            avg_psnr = sum(baseline_info['psnr_values'][-5:]) / min(5, len(baseline_info['psnr_values']))
            report.append(f"平均PSNR (最后5个): {avg_psnr:.2f} dB")
        
        if baseline_ckpt and baseline_ckpt['model_size'] > 0:
            report.append(f"模型大小: {baseline_ckpt['model_size']:.2f} MB")
    else:
        report.append("❌ 未找到baseline结果")
    
    report.append("")
    
    # 网格剪枝结果
    report.append("-" * 80)
    report.append("测试2: 使用网格剪枝 (Instant4D)")
    report.append("-" * 80)
    
    if pruning_info:
        report.append(f"日志文件: {pruning_info['log_path']}")
        
        if pruning_info['pruning_info']:
            pi = pruning_info['pruning_info']
            report.append("")
            report.append("网格剪枝详情:")
            if 'original_points' in pi:
                report.append(f"  原始点云: {pi['original_points']:,} 个点")
            if 'pruned_points' in pi:
                report.append(f"  剪枝后:   {pi['pruned_points']:,} 个点")
            if 'reduction_ratio' in pi:
                report.append(f"  减少比例: {pi['reduction_ratio']:.1f}%")
            if 'voxel_size' in pi:
                report.append(f"  体素大小: {pi['voxel_size']:.6f}")
        
        report.append("")
        report.append(f"初始点云数量: {pruning_info['initial_points']:,}" if pruning_info['initial_points'] else "初始点云数量: 未知")
        
        if pruning_info['psnr_values']:
            avg_psnr = sum(pruning_info['psnr_values'][-5:]) / min(5, len(pruning_info['psnr_values']))
            report.append(f"平均PSNR (最后5个): {avg_psnr:.2f} dB")
        
        if pruning_ckpt and pruning_ckpt['model_size'] > 0:
            report.append(f"模型大小: {pruning_ckpt['model_size']:.2f} MB")
    else:
        report.append("❌ 未找到网格剪枝结果")
    
    report.append("")
    
    # 性能对比
    if baseline_info and pruning_info:
        report.append("-" * 80)
        report.append("性能对比分析")
        report.append("-" * 80)
        
        comparison = compare_results(baseline_info, pruning_info)
        
        if comparison['point_reduction'] is not None:
            report.append(f"✓ 点云数量减少: {comparison['point_reduction']:.1f}%")
        
        if baseline_ckpt and pruning_ckpt:
            if baseline_ckpt['model_size'] > 0 and pruning_ckpt['model_size'] > 0:
                size_reduction = (1 - pruning_ckpt['model_size'] / baseline_ckpt['model_size']) * 100
                report.append(f"✓ 模型大小减少: {size_reduction:.1f}%")
        
        if comparison['psnr_diff'] is not None:
            if comparison['psnr_diff'] > 0:
                report.append(f"✓ PSNR提升: +{comparison['psnr_diff']:.2f} dB")
            else:
                report.append(f"⚠ PSNR变化: {comparison['psnr_diff']:.2f} dB")
        
        report.append("")
    
    # 与论文对比
    report.append("-" * 80)
    report.append("与 Instant4D 论文预期对比")
    report.append("-" * 80)
    report.append("论文预期效果:")
    report.append("  - 点云减少: ~92%")
    report.append("  - 训练加速: ~4x")
    report.append("  - 渲染提升: ~6x")
    report.append("  - 内存减少: ~90%")
    report.append("  - PSNR提升: +0.8 dB")
    report.append("")
    
    if baseline_info and pruning_info:
        comparison = compare_results(baseline_info, pruning_info)
        report.append("本次测试结果:")
        if comparison['point_reduction'] is not None:
            report.append(f"  - 点云减少: {comparison['point_reduction']:.1f}%")
        if comparison['psnr_diff'] is not None:
            report.append(f"  - PSNR变化: {comparison['psnr_diff']:+.2f} dB")
        report.append("")
        report.append("注: Debug模式迭代较少，与论文完整训练有差异")
    
    report.append("")
    report.append("=" * 80)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def main():
    """主函数"""
    print("=" * 80)
    print("网格剪枝 Debug 结果分析工具")
    print("=" * 80)
    print()
    
    # 定义路径
    debug_results_dir = "debug_results"
    
    baseline_log = os.path.join(debug_results_dir, "log_no_pruning.txt")
    pruning_log = os.path.join(debug_results_dir, "log_with_pruning.txt")
    
    baseline_ckpt_dir = "output/debug/sear_steak_no_pruning/point_cloud/iteration_500"
    pruning_ckpt_dir = "output/debug/sear_steak_with_pruning/point_cloud/iteration_500"
    
    # 解析日志
    print("📊 解析训练日志...")
    baseline_info = parse_log_file(baseline_log)
    pruning_info = parse_log_file(pruning_log)
    
    # 分析检查点
    print("📦 分析模型检查点...")
    baseline_ckpt = analyze_checkpoint(baseline_ckpt_dir)
    pruning_ckpt = analyze_checkpoint(pruning_ckpt_dir)
    
    # 生成报告
    print("📝 生成分析报告...")
    report_path = os.path.join(debug_results_dir, "detailed_analysis.txt")
    os.makedirs(debug_results_dir, exist_ok=True)
    
    report = generate_report(baseline_info, pruning_info, baseline_ckpt, pruning_ckpt, report_path)
    
    print()
    print(report)
    print()
    print(f"✓ 详细报告已保存: {report_path}")
    print()
    
    # 输出文件位置摘要
    print("-" * 80)
    print("📁 输出文件位置摘要")
    print("-" * 80)
    print(f"分析报告:     {report_path}")
    print(f"Baseline日志: {baseline_log}")
    print(f"网格剪枝日志: {pruning_log}")
    print(f"Baseline模型: {baseline_ckpt_dir}")
    print(f"网格剪枝模型: {pruning_ckpt_dir}")
    print()
    
    # 可视化结果位置
    print("-" * 80)
    print("🎨 可视化结果位置")
    print("-" * 80)
    print("Baseline渲染结果:")
    print("  - output/debug/sear_steak_no_pruning/test/ours_500/")
    print("  - output/debug/sear_steak_no_pruning/video/")
    print()
    print("网格剪枝渲染结果:")
    print("  - output/debug/sear_steak_with_pruning/test/ours_500/")
    print("  - output/debug/sear_steak_with_pruning/video/")
    print("=" * 80)

if __name__ == "__main__":
    main()

