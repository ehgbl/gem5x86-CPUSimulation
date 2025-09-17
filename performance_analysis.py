
"""
Performance Analysis and Visualization Script
=============================================

This script provides advanced analysis and visualization capabilities
for gem5 simulation results, including performance comparisons,
statistical analysis, and interactive plots.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class PerformanceAnalyzer:
    """Advanced performance analysis and visualization for gem5 simulations."""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        

        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def parse_gem5_stats(self, stats_file):
        """Parse gem5 statistics file and extract performance metrics."""
        stats = {}
        
        try:
            with open(stats_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        

                        try:
                            if '.' in value:
                                stats[key] = float(value)
                            else:
                                stats[key] = int(value)
                        except ValueError:
                            stats[key] = value
                            
        except FileNotFoundError:
            print(f"Warning: Stats file {stats_file} not found")
            
        return stats
    
    def calculate_cache_metrics(self, stats):
        """Calculate comprehensive cache performance metrics."""
        metrics = {}
        

        l1i_hits = stats.get('system.cpu_icache.hits', 0)
        l1i_misses = stats.get('system.cpu_icache.misses', 0)
        l1i_total = l1i_hits + l1i_misses
        
        metrics['l1i_hit_rate'] = (l1i_hits / l1i_total * 100) if l1i_total > 0 else 0
        metrics['l1i_miss_rate'] = (l1i_misses / l1i_total * 100) if l1i_total > 0 else 0
        metrics['l1i_hits'] = l1i_hits
        metrics['l1i_misses'] = l1i_misses
        

        l1d_hits = stats.get('system.cpu_dcache.hits', 0)
        l1d_misses = stats.get('system.cpu_dcache.misses', 0)
        l1d_total = l1d_hits + l1d_misses
        
        metrics['l1d_hit_rate'] = (l1d_hits / l1d_total * 100) if l1d_total > 0 else 0
        metrics['l1d_miss_rate'] = (l1d_misses / l1d_total * 100) if l1d_total > 0 else 0
        metrics['l1d_hits'] = l1d_hits
        metrics['l1d_misses'] = l1d_misses
        

        l2_hits = stats.get('system.l2cache.hits', 0)
        l2_misses = stats.get('system.l2cache.misses', 0)
        l2_total = l2_hits + l2_misses
        
        metrics['l2_hit_rate'] = (l2_hits / l2_total * 100) if l2_total > 0 else 0
        metrics['l2_miss_rate'] = (l2_misses / l2_total * 100) if l2_total > 0 else 0
        metrics['l2_hits'] = l2_hits
        metrics['l2_misses'] = l2_misses
        
        return metrics
    
    def calculate_memory_metrics(self, stats):
        """Calculate memory subsystem performance metrics."""
        metrics = {}
        

        metrics['mem_reads'] = stats.get('system.mem_ctrl.num_reads', 0)
        metrics['mem_writes'] = stats.get('system.mem_ctrl.num_writes', 0)
        metrics['mem_read_bytes'] = stats.get('system.mem_ctrl.bytes_read', 0)
        metrics['mem_write_bytes'] = stats.get('system.mem_ctrl.bytes_written', 0)
        

        metrics['avg_mem_latency'] = stats.get('system.mem_ctrl.avg_memory_latency', 0)
        metrics['mem_bw_util'] = stats.get('system.mem_ctrl.bw_util', 0)
        

        total_bytes = metrics['mem_read_bytes'] + metrics['mem_write_bytes']
        sim_ticks = stats.get('sim_ticks', 1)
        sim_time = sim_ticks / 1e12  # Convert to seconds
        metrics['memory_bandwidth'] = total_bytes / sim_time / (1024**2)  # MB/s
        
        return metrics
    
    def calculate_cpu_metrics(self, stats, num_cores=1):
        """Calculate CPU performance metrics."""
        metrics = {}
        
        total_insts = 0
        total_cycles = 0
        
        for i in range(num_cores):
            insts = stats.get(f'system.cpu{i}.numInsts', 0)
            cycles = stats.get(f'system.cpu{i}.numCycles', 0)
            
            total_insts += insts
            total_cycles += cycles
            
            metrics[f'cpu{i}_insts'] = insts
            metrics[f'cpu{i}_cycles'] = cycles
            metrics[f'cpu{i}_cpi'] = cycles / insts if insts > 0 else 0
            metrics[f'cpu{i}_ipc'] = insts / cycles if cycles > 0 else 0
        
        metrics['total_insts'] = total_insts
        metrics['total_cycles'] = total_cycles
        metrics['avg_cpi'] = total_cycles / total_insts if total_insts > 0 else 0
        metrics['avg_ipc'] = total_insts / total_cycles if total_cycles > 0 else 0
        
        return metrics
    
    def create_cache_visualization(self, metrics, save_path=None):
        """Create comprehensive cache performance visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cache Performance Analysis', fontsize=16, fontweight='bold')
        

        cache_levels = ['L1I', 'L1D', 'L2']
        hit_rates = [metrics['l1i_hit_rate'], metrics['l1d_hit_rate'], metrics['l2_hit_rate']]
        
        axes[0, 0].bar(cache_levels, hit_rates, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Cache Hit Rates')
        axes[0, 0].set_ylabel('Hit Rate (%)')
        axes[0, 0].set_ylim(0, 100)
        

        for i, v in enumerate(hit_rates):
            axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        

        l1i_total = metrics['l1i_hits'] + metrics['l1i_misses']
        l1d_total = metrics['l1d_hits'] + metrics['l1d_misses']
        l2_total = metrics['l2_hits'] + metrics['l2_misses']
        
        sizes = [l1i_total, l1d_total, l2_total]
        labels = ['L1I', 'L1D', 'L2']
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Cache Access Distribution')
        

        miss_rates = [metrics['l1i_miss_rate'], metrics['l1d_miss_rate'], metrics['l2_miss_rate']]
        
        axes[1, 0].bar(cache_levels, miss_rates, color=['salmon', 'tomato', 'darkred'])
        axes[1, 0].set_title('Cache Miss Rates')
        axes[1, 0].set_ylabel('Miss Rate (%)')
        axes[1, 0].set_ylim(0, max(miss_rates) * 1.1)
        

        for i, v in enumerate(miss_rates):
            axes[1, 0].text(i, v + max(miss_rates) * 0.01, f'{v:.1f}%', ha='center', va='bottom')
        

        efficiency_data = np.array([
            [metrics['l1i_hit_rate'], metrics['l1i_miss_rate']],
            [metrics['l1d_hit_rate'], metrics['l1d_miss_rate']],
            [metrics['l2_hit_rate'], metrics['l2_miss_rate']]
        ])
        
        im = axes[1, 1].imshow(efficiency_data, cmap='RdYlGn', aspect='auto')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_xticklabels(['Hit Rate', 'Miss Rate'])
        axes[1, 1].set_yticks([0, 1, 2])
        axes[1, 1].set_yticklabels(['L1I', 'L1D', 'L2'])
        axes[1, 1].set_title('Cache Efficiency Heatmap')
        

        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_memory_visualization(self, metrics, save_path=None):
        """Create memory performance visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Memory Performance Analysis', fontsize=16, fontweight='bold')
        

        operations = ['Reads', 'Writes']
        counts = [metrics['mem_reads'], metrics['mem_writes']]
        
        axes[0, 0].bar(operations, counts, color=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Memory Operations')
        axes[0, 0].set_ylabel('Number of Operations')
        

        for i, v in enumerate(counts):
            axes[0, 0].text(i, v + max(counts) * 0.01, f'{v:,}', ha='center', va='bottom')
        

        read_mb = metrics['mem_read_bytes'] / (1024**2)
        write_mb = metrics['mem_write_bytes'] / (1024**2)
        
        axes[0, 1].bar(['Read', 'Write'], [read_mb, write_mb], color=['lightgreen', 'orange'])
        axes[0, 1].set_title('Data Transfer')
        axes[0, 1].set_ylabel('Data (MB)')
        

        for i, v in enumerate([read_mb, write_mb]):
            axes[0, 1].text(i, v + max([read_mb, write_mb]) * 0.01, f'{v:.1f} MB', ha='center', va='bottom')
        

        latency = metrics['avg_mem_latency']
        axes[1, 0].bar(['Average'], [latency], color='gold')
        axes[1, 0].set_title('Memory Latency')
        axes[1, 0].set_ylabel('Cycles')
        axes[1, 0].text(0, latency + latency * 0.01, f'{latency:.1f} cycles', ha='center', va='bottom')
        

        bw_util = metrics['mem_bw_util']
        axes[1, 1].bar(['Utilization'], [bw_util], color='purple')
        axes[1, 1].set_title('Memory Bandwidth Utilization')
        axes[1, 1].set_ylabel('Utilization (%)')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].text(0, bw_util + 1, f'{bw_util:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_cpu_visualization(self, metrics, num_cores=1, save_path=None):
        """Create CPU performance visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CPU Performance Analysis', fontsize=16, fontweight='bold')
        

        cores = list(range(num_cores))
        ipc_values = [metrics.get(f'cpu{i}_ipc', 0) for i in cores]
        
        axes[0, 0].bar(cores, ipc_values, color='lightblue')
        axes[0, 0].set_title('Instructions per Cycle (IPC)')
        axes[0, 0].set_xlabel('CPU Core')
        axes[0, 0].set_ylabel('IPC')
        

        for i, v in enumerate(ipc_values):
            axes[0, 0].text(i, v + max(ipc_values) * 0.01, f'{v:.2f}', ha='center', va='bottom')
        

        cpi_values = [metrics.get(f'cpu{i}_cpi', 0) for i in cores]
        
        axes[0, 1].bar(cores, cpi_values, color='lightcoral')
        axes[0, 1].set_title('Cycles per Instruction (CPI)')
        axes[0, 1].set_xlabel('CPU Core')
        axes[0, 1].set_ylabel('CPI')
        

        for i, v in enumerate(cpi_values):
            axes[0, 1].text(i, v + max(cpi_values) * 0.01, f'{v:.2f}', ha='center', va='bottom')
        

        inst_values = [metrics.get(f'cpu{i}_insts', 0) for i in cores]
        
        axes[1, 0].bar(cores, inst_values, color='lightgreen')
        axes[1, 0].set_title('Instructions Executed')
        axes[1, 0].set_xlabel('CPU Core')
        axes[1, 0].set_ylabel('Instructions')
        

        for i, v in enumerate(inst_values):
            axes[1, 0].text(i, v + max(inst_values) * 0.01, f'{v:,}', ha='center', va='bottom')
        

        performance_data = {
            'IPC': ipc_values,
            'CPI': cpi_values
        }
        
        x = np.arange(len(cores))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, ipc_values, width, label='IPC', color='skyblue')
        axes[1, 1].bar(x + width/2, cpi_values, width, label='CPI', color='salmon')
        
        axes[1, 1].set_title('Performance Comparison')
        axes[1, 1].set_xlabel('CPU Core')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(cores)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, metrics, save_path=None):
        """Create interactive Plotly dashboard."""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cache Hit Rates', 'Memory Performance', 'CPU Performance', 'System Overview'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        

        cache_levels = ['L1I', 'L1D', 'L2']
        hit_rates = [metrics['l1i_hit_rate'], metrics['l1d_hit_rate'], metrics['l2_hit_rate']]
        
        fig.add_trace(
            go.Bar(x=cache_levels, y=hit_rates, name='Hit Rate', marker_color='lightblue'),
            row=1, col=1
        )
        

        operations = ['Reads', 'Writes']
        counts = [metrics['mem_reads'], metrics['mem_writes']]
        
        fig.add_trace(
            go.Bar(x=operations, y=counts, name='Operations', marker_color='lightcoral'),
            row=1, col=2
        )
        

        cpu_metrics = ['IPC', 'CPI']
        cpu_values = [metrics.get('cpu0_ipc', 0), metrics.get('cpu0_cpi', 0)]
        
        fig.add_trace(
            go.Bar(x=cpu_metrics, y=cpu_values, name='CPU Metrics', marker_color='lightgreen'),
            row=2, col=1
        )
        

        overall_performance = (metrics['l1i_hit_rate'] + metrics['l1d_hit_rate'] + metrics['l2_hit_rate']) / 3
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_performance,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Cache Performance"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Gem5 Simulation Performance Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def generate_report(self, stats_file, output_dir='results'):
        """Generate comprehensive performance report."""
        print("Generating performance analysis report...")
        

        stats = self.parse_gem5_stats(stats_file)
        

        cache_metrics = self.calculate_cache_metrics(stats)
        memory_metrics = self.calculate_memory_metrics(stats)
        cpu_metrics = self.calculate_cpu_metrics(stats)
        

        all_metrics = {**cache_metrics, **memory_metrics, **cpu_metrics}
        

        self.create_cache_visualization(cache_metrics, 
                                      os.path.join(output_dir, 'cache_analysis.png'))
        self.create_memory_visualization(memory_metrics,
                                       os.path.join(output_dir, 'memory_analysis.png'))
        self.create_cpu_visualization(cpu_metrics,
                                    os.path.join(output_dir, 'cpu_analysis.png'))
        

        self.create_interactive_dashboard(all_metrics,
                                        os.path.join(output_dir, 'performance_dashboard.html'))
        

        with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"Performance analysis complete! Results saved to {output_dir}")
        
        return all_metrics


def main():
    """Main function for performance analysis."""
    parser = argparse.ArgumentParser(description='Gem5 Performance Analysis Tool')
    parser.add_argument('--stats-file', default='stats.txt',
                       help='Path to gem5 statistics file')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for analysis results')
    parser.add_argument('--interactive', action='store_true',
                       help='Show interactive plots')
    
    args = parser.parse_args()
    

    analyzer = PerformanceAnalyzer(args.output_dir)
    

    metrics = analyzer.generate_report(args.stats_file, args.output_dir)
    

    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"L1I Hit Rate: {metrics['l1i_hit_rate']:.2f}%")
    print(f"L1D Hit Rate: {metrics['l1d_hit_rate']:.2f}%")
    print(f"L2 Hit Rate: {metrics['l2_hit_rate']:.2f}%")
    print(f"Average IPC: {metrics['avg_ipc']:.2f}")
    print(f"Memory Bandwidth: {metrics['memory_bandwidth']:.2f} MB/s")
    print(f"Memory Latency: {metrics['avg_mem_latency']:.2f} cycles")


if __name__ == '__main__':
    main()
