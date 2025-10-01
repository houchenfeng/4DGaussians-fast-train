import time
import json
import os

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
        
    def start_timer(self, name):
        """开始计时一个特定的操作"""
        self.timers[name] = {'start': time.time(), 'elapsed': 0, 'count': 0}
        
    def end_timer(self, name):
        """结束计时一个特定的操作"""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]['start']
            self.timers[name]['elapsed'] += elapsed
            self.timers[name]['count'] += 1
            
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
        
    def save_timing_report(self, filename="timing_report.json"):
        """保存计时报告到文件"""
        if self.output_dir:
            report = {
                'total_training_time': self.get_total_time(),
                'detailed_timings': {}
            }
            
            for name, data in self.timers.items():
                report['detailed_timings'][name] = {
                    'total_elapsed': data['elapsed'],
                    'call_count': data['count'],
                    'average_per_call': data['elapsed'] / max(data['count'], 1)
                }
                
            report_path = os.path.join(self.output_dir, filename)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"计时报告已保存到: {report_path}")
            
    def print_summary(self):
        """打印计时摘要"""
        print("\n=== 计时摘要 ===")
        print(f"总训练时间: {self.get_total_time():.2f} 秒")
        for name, data in self.timers.items():
            avg_time = data['elapsed'] / max(data['count'], 1)
            print(f"{name}: 总时间 {data['elapsed']:.2f}s, 调用次数 {data['count']}, 平均每次 {avg_time:.4f}s")