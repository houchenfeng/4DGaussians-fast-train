import time
import json
import os
from datetime import datetime

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
        self.paused = False

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()
        elif self.paused:
            self.start_time = time.time() - self.elapsed
            self.paused = False

    def pause(self):
        if not self.paused:
            self.elapsed = time.time() - self.start_time
            self.paused = True

    def get_elapsed_time(self):
        if self.paused:
            return self.elapsed
        else:
            return time.time() - self.start_time

class DetailedTimer:
    def __init__(self, output_dir=None):
        self.timers = {}
        self.start_time = time.time()
        self.output_dir = output_dir
        self.total_start = time.time()
        self.training_logs = []  # 存储训练日志
        self.current_iteration = 0
        self.per_iteration_times = {}  # 存储每轮单独用时
        self.last_iteration_end_time = time.time()  # 记录上一轮结束时间
        self.current_iteration_start_time = time.time()  # 记录当前轮次开始时间
        self.current_iteration_timers = {}  # 存储当前轮次各操作的累积时间
        
    def start_iteration(self, iteration):
        """开始一轮迭代的计时"""
        self.current_iteration = iteration
        self.current_iteration_start_time = time.time()
        # 初始化当前轮次的计时器
        self.current_iteration_timers = {}
        
    def start_timer(self, name):
        """开始计时一个特定的操作"""
        if name not in self.timers:
            self.timers[name] = {'start': 0, 'elapsed': 0, 'count': 0}
        self.timers[name]['start'] = time.time()
        
    def end_timer(self, name):
        """结束计时一个特定的操作"""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]['start']
            self.timers[name]['elapsed'] += elapsed
            self.timers[name]['count'] += 1
            # 记录当前轮次的操作用时
            if name not in self.current_iteration_timers:
                self.current_iteration_timers[name] = 0
            self.current_iteration_timers[name] += elapsed
            
    def record_iteration_timing(self, iteration, stage):
        """记录每轮单独用时"""
        current_time = time.time()
        
        if iteration not in self.per_iteration_times:
            self.per_iteration_times[iteration] = {}
            
        # 计算这轮的总用时（从当前轮次开始到结束）
        iteration_total_time = current_time - self.current_iteration_start_time
        self.per_iteration_times[iteration][f'{stage}_total'] = iteration_total_time
        
        # 记录各个子操作的单独用时（使用当前轮次的计时器）
        for timer_name, timer_data in self.timers.items():
            if timer_name.startswith(stage) and timer_name in self.current_iteration_timers:
                iteration_operation_time = self.current_iteration_timers[timer_name]
                self.per_iteration_times[iteration][timer_name] = iteration_operation_time
        
        # 更新当前轮次开始时间（为下一轮准备）
        # 注意：这里不更新current_iteration_start_time，因为它应该在下一轮开始时更新
        
    def get_iteration_timing(self, iteration, stage):
        """获取指定轮次的用时信息"""
        if iteration not in self.per_iteration_times:
            return None
        return self.per_iteration_times[iteration]
            
    def pause_timer(self, name):
        """暂停计时"""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]['start']
            self.timers[name]['elapsed'] += elapsed
            
    def resume_timer(self, name):
        """恢复计时"""
        if name in self.timers:
            self.timers[name]['start'] = time.time()
            
    def get_total_time(self):
        """获取总训练时间"""
        return time.time() - self.total_start
        
    def log_iteration(self, iteration, loss, psnr=None, l1_loss=None, stage="", **kwargs):
        """记录每轮训练的详细信息"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = self.get_total_time()
        
        # 确保所有数值都转换为 Python 原生类型
        def convert_to_serializable(obj):
            """将对象转换为 JSON 可序列化的类型"""
            import torch
            if torch.is_tensor(obj):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                # 尝试转换为字符串
                return str(obj)
        
        log_entry = {
            'timestamp': current_time,
            'iteration': iteration,
            'stage': stage,
            'elapsed_time_seconds': elapsed_time,
            'loss': convert_to_serializable(loss),
            'psnr': convert_to_serializable(psnr),
            'l1_loss': convert_to_serializable(l1_loss),
        }
        
        # 处理额外的参数
        for key, value in kwargs.items():
            log_entry[key] = convert_to_serializable(value)
        
        # 记录每轮单独用时
        self.record_iteration_timing(iteration, stage)
        
        # 添加每轮用时信息到日志
        if iteration in self.per_iteration_times:
            iteration_timings = self.per_iteration_times[iteration]
            log_entry['iteration_timings'] = iteration_timings
            log_entry['iteration_total_time'] = iteration_timings.get(f'{stage}_total', 0)
        
        self.training_logs.append(log_entry)
        self.current_iteration = iteration
        
        # 打印当前轮次信息
        iteration_total = log_entry.get('iteration_total_time', 0)
        print(f"\n[{current_time}] 迭代 {iteration} ({stage}) - 累计用时: {elapsed_time:.1f}s, 本轮用时: {iteration_total:.3f}s")
        print(f"  损失: {loss:.6f}" + 
              (f", PSNR: {psnr:.2f}" if psnr is not None else "") + 
              (f", L1: {l1_loss:.6f}" if l1_loss is not None else ""))
        
        # 打印当前轮次详细计时
        self.print_iteration_timings(iteration, stage)
        
    def print_iteration_timings(self, iteration, stage):
        """打印当前轮次的详细计时"""
        if iteration not in self.per_iteration_times:
            return
            
        print(f"  第{iteration}轮详细用时 ({stage}):")
        iteration_timings = self.per_iteration_times[iteration]
        
        # 显示本轮总用时
        total_time = iteration_timings.get(f'{stage}_total', 0)
        print(f"    本轮总用时: {total_time:.3f}s")
        
        # 显示各个子操作的用时
        for timer_name, time_val in iteration_timings.items():
            if timer_name != f'{stage}_total' and timer_name.startswith(stage):
                operation_name = timer_name.replace(f'{stage}_', '')
                percentage = (time_val / total_time * 100) if total_time > 0 else 0
                print(f"    {operation_name}: {time_val:.3f}s ({percentage:.1f}%)")
        
    def print_current_timings(self):
        """打印当前计时器状态"""
        print("  当前计时状态:")
        total_time = self.get_total_time()
        for name, data in self.timers.items():
            if data['elapsed'] > 0:
                percentage = (data['elapsed'] / total_time) * 100
                print(f"    {name}: {data['elapsed']:.3f}s ({percentage:.1f}%)")
        
    def save_timing_report(self, filename="timing_report.json"):
        """保存计时报告到文件"""
        if self.output_dir:
            # 确保所有数据都是 JSON 可序列化的
            def ensure_serializable(obj):
                """确保对象是 JSON 可序列化的"""
                import torch
                if torch.is_tensor(obj):
                    return obj.item() if obj.numel() == 1 else obj.tolist()
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [ensure_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: ensure_serializable(value) for key, value in obj.items()}
                else:
                    return str(obj)
            
            # 分离迭代计时器和子计时器，避免重复计算
            iteration_timers = {}
            sub_timers = {}
            total_sub_time = 0
            
            for name, data in self.timers.items():
                if data['elapsed'] > 0:
                    if name.endswith('_iteration'):
                        iteration_timers[name] = data
                    else:
                        sub_timers[name] = data
                        total_sub_time += data['elapsed']
            
            report = {
                'total_training_time': self.get_total_time(),
                'training_logs': ensure_serializable(self.training_logs),
                'per_iteration_timings': ensure_serializable(self.per_iteration_times),
                'detailed_timings': {
                    'iteration_timers': {},
                    'sub_timers': {},
                    'unaccounted_time': 0
                }
            }
            
            # 计算未计时的部分
            total_time = self.get_total_time()
            unaccounted_time = total_time - total_sub_time
            
            # 保存迭代计时器
            for name, data in iteration_timers.items():
                report['detailed_timings']['iteration_timers'][name] = {
                    'total_elapsed': data['elapsed'],
                    'percentage': (data['elapsed'] / total_time) * 100
                }
            
            # 保存子计时器
            for name, data in sub_timers.items():
                report['detailed_timings']['sub_timers'][name] = {
                    'total_elapsed': data['elapsed'],
                    'percentage': (data['elapsed'] / total_time) * 100
                }
            
            # 添加未计时的部分
            if unaccounted_time > 0:
                report['detailed_timings']['unaccounted_time'] = {
                    'total_elapsed': unaccounted_time,
                    'percentage': (unaccounted_time / total_time) * 100,
                    'description': '未计时部分（初始化、清理、等待等）'
                }
                
            report_path = os.path.join(self.output_dir, filename)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"计时报告已保存到: {report_path}")
            
    def save_training_logs(self, filename="training_logs.json"):
        """单独保存训练日志"""
        if self.output_dir and self.training_logs:
            # 确保所有数据都是 JSON 可序列化的
            def ensure_serializable(obj):
                """确保对象是 JSON 可序列化的"""
                import torch
                if torch.is_tensor(obj):
                    return obj.item() if obj.numel() == 1 else obj.tolist()
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                elif isinstance(obj, (list, tuple)):
                    return [ensure_serializable(item) for item in obj]
                elif isinstance(obj, dict):
                    return {key: ensure_serializable(value) for key, value in obj.items()}
                else:
                    return str(obj)
            
            log_path = os.path.join(self.output_dir, filename)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(ensure_serializable(self.training_logs), f, indent=2, ensure_ascii=False)
            print(f"训练日志已保存到: {log_path}")
            
    def print_summary(self):
        """打印计时摘要"""
        print("\n=== 计时摘要 ===")
        total_time = self.get_total_time()
        print(f"总训练时间: {total_time:.2f} 秒")
        print(f"总迭代次数: {self.current_iteration}")
        
        # 分离迭代计时器和子计时器
        iteration_timers = {}
        sub_timers = {}
        total_sub_time = 0
        
        for name, data in self.timers.items():
            if data['elapsed'] > 0:
                if name.endswith('_iteration'):
                    iteration_timers[name] = data
                else:
                    sub_timers[name] = data
                    total_sub_time += data['elapsed']
        
        # 计算未计时部分
        unaccounted_time = total_time - total_sub_time
        
        print("\n迭代阶段耗时:")
        for name, data in iteration_timers.items():
            percentage = (data['elapsed'] / total_time) * 100
            print(f"  {name}: {data['elapsed']:.2f}s ({percentage:.1f}%)")
        
        print("\n子操作耗时:")
        for name, data in sub_timers.items():
            percentage = (data['elapsed'] / total_time) * 100
            print(f"  {name}: {data['elapsed']:.2f}s ({percentage:.1f}%)")
        
        if unaccounted_time > 0:
            percentage = (unaccounted_time / total_time) * 100
            print(f"\n未计时部分: {unaccounted_time:.2f}s ({percentage:.1f}%)")
            print("  (初始化、清理、等待等)")
        
        # 验证百分比总和
        total_percentage = (total_sub_time / total_time) * 100 + (unaccounted_time / total_time) * 100
        print(f"\n验证: 子操作 + 未计时 = {total_percentage:.1f}%")
                
        # 打印训练日志摘要
        if self.training_logs:
            print(f"\n训练日志条目数: {len(self.training_logs)}")
            if len(self.training_logs) > 0:
                final_log = self.training_logs[-1]
                print(f"最终损失: {final_log.get('loss', 'N/A')}")
                if 'psnr' in final_log and final_log['psnr'] is not None:
                    print(f"最终PSNR: {final_log['psnr']:.2f}")