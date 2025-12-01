"""
Visualize Query Execution Performance Metrics
Generates graphs from logs/query_execution_metrics.jsonl
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
import statistics

plt.style.use('seaborn-v0_8-darkgrid')

class QueryMetricsVisualizer:
    """Visualize query execution performance metrics"""
    
    def __init__(self, metrics_file='logs/query_execution_metrics.jsonl'):
        self.metrics_file = Path(metrics_file)
        self.output_dir = Path("performance/graphs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = self.load_metrics()
    
    def load_metrics(self):
        """Load metrics from JSONL file"""
        metrics = []
        if not self.metrics_file.exists():
            print(f"⚠️  Warning: {self.metrics_file} not found")
            return metrics
        
        with open(self.metrics_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        metrics.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        print(f"📊 Loaded {len(metrics)} query execution records")
        return metrics
    
    def create_response_time_histogram(self):
        """Create histogram of response times"""
        if not self.metrics:
            print("⚠️  No metrics data for histogram")
            return
        
        print("📊 Creating response time histogram...")
        
        response_times = [m.get('response_time_seconds', 0) * 1000 for m in self.metrics]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        n, bins, patches = ax.hist(response_times, bins=20, color='#2ecc71', 
                                   alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Response Time (ms)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title('Query Response Time Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "response_time_histogram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_response_time_timeline(self):
        """Create timeline of response times"""
        if not self.metrics:
            print("⚠️  No metrics data for timeline")
            return
        
        print("📊 Creating response time timeline...")
        
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in self.metrics]
        response_times = [m.get('response_time_seconds', 0) * 1000 for m in self.metrics]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot all dots in green
        ax.scatter(timestamps, response_times, c='#2ecc71', s=100, alpha=0.6, edgecolors='black')
        # ax.plot(timestamps, response_times, color='gray', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Date', fontsize=14, fontweight='bold')
        ax.set_ylabel('Response Time (ms)', fontsize=14, fontweight='bold')
        ax.set_title('Query Response Time Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        output_path = self.output_dir / "response_time_timeline.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_entities_triples_chart(self):
        """Create chart showing entities found and triples extracted"""
        if not self.metrics:
            print("⚠️  No metrics data for entities/triples chart")
            return
        
        print("📊 Creating entities and triples chart...")
        
        entities = [m.get('entities_found', 0) for m in self.metrics]
        triples = [m.get('triples_extracted', 0) for m in self.metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Entities histogram
        ax1.hist(entities, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Entities Found', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title(f'Entities Found Distribution\n(Mean: {statistics.mean(entities):.1f})', 
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Triples histogram
        ax2.hist(triples, bins=15, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Triples Extracted', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title(f'Triples Extracted Distribution\n(Mean: {statistics.mean(triples):.1f})', 
                     fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Knowledge Graph Extraction Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        output_path = self.output_dir / "entities_triples_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_performance_summary_dashboard(self):
        """Create comprehensive performance summary dashboard"""
        if not self.metrics:
            print("⚠️  No metrics data for dashboard")
            return
        
        print("📊 Creating performance summary dashboard...")
        
        response_times = [m.get('response_time_seconds', 0) * 1000 for m in self.metrics]
        entities = [m.get('entities_found', 0) for m in self.metrics]
        triples = [m.get('triples_extracted', 0) for m in self.metrics]
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Performance statistics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        
        stats_text = (
            "QUERY PERFORMANCE STATISTICS\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Total Queries: {len(self.metrics)}\n\n"
            "Response Time:\n"
            f"  Mean:     {statistics.mean(response_times):>8.2f} ms\n"
            f"  Median:   {statistics.median(response_times):>8.2f} ms\n"
            f"  Min:      {min(response_times):>8.2f} ms\n"
            f"  Max:      {max(response_times):>8.2f} ms\n"
            f"  Std Dev:  {statistics.stdev(response_times):>8.2f} ms\n\n"
            "Entities Found:\n"
            f"  Mean:     {statistics.mean(entities):>8.1f}\n"
            f"  Total:    {sum(entities):>8}\n\n"
            "Triples Extracted:\n"
            f"  Mean:     {statistics.mean(triples):>8.1f}\n"
            f"  Total:    {sum(triples):>8}\n"
        )
        
        ax1.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax1.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=10)
        
        # 2. Performance percentiles (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        sorted_times = sorted(response_times)
        percentiles = [50, 75, 90, 95, 99]
        percentile_values = [sorted_times[len(sorted_times) * p // 100] for p in percentiles]
        
        bars = ax2.barh(percentiles, percentile_values, color='#3498db', alpha=0.7, edgecolor='black')
        for i, (p, val) in enumerate(zip(percentiles, percentile_values)):
            ax2.text(val + 50, i, f'{val:.1f}ms', va='center', fontweight='bold')
        
        ax2.set_xlabel('Response Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Percentile', fontsize=12, fontweight='bold')
        ax2.set_title('Response Time Percentiles', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Response time distribution (middle row, full width)
        ax3 = fig.add_subplot(gs[1, :])
        n, bins, patches = ax3.hist(response_times, bins=25, color='#3498db', 
                                    alpha=0.7, edgecolor='black')
        
        # Color bars by performance
        for i, patch in enumerate(patches):
            if bins[i] < 500:
                patch.set_facecolor('#2ecc71')
            elif bins[i] < 1000:
                patch.set_facecolor('#3498db')
            elif bins[i] < 2000:
                patch.set_facecolor('#f39c12')
            else:
                patch.set_facecolor('#e74c3c')
        
        ax3.axvline(statistics.mean(response_times), color='red', linestyle='--', 
                   linewidth=2, label='Mean')
        ax3.axvline(statistics.median(response_times), color='green', linestyle='--', 
                   linewidth=2, label='Median')
        
        ax3.set_xlabel('Response Time (ms)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Response Time Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Entities vs Triples scatter (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.scatter(entities, triples, c='#9b59b6', s=100, alpha=0.6, edgecolors='black')
        
        # Add trend line
        z = np.polyfit(entities, triples, 1)
        p = np.poly1d(z)
        ax4.plot(sorted(entities), p(sorted(entities)), "r--", alpha=0.8, linewidth=2)
        
        ax4.set_xlabel('Entities Found', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Triples Extracted', fontsize=12, fontweight='bold')
        ax4.set_title('Entities vs Triples Correlation', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance rating gauge (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        mean_time = statistics.mean(response_times)
        
        categories = ['Excellent\n(<500ms)', 'Good\n(500-1000ms)', 
                     'Acceptable\n(1000-2000ms)', 'Poor\n(>2000ms)']
        counts = [
            sum(1 for t in response_times if t < 500),
            sum(1 for t in response_times if 500 <= t < 1000),
            sum(1 for t in response_times if 1000 <= t < 2000),
            sum(1 for t in response_times if t >= 2000)
        ]
        colors_gauge = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        
        bars = ax5.barh(categories, counts, color=colors_gauge, alpha=0.7, edgecolor='black')
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            percentage = (count / len(response_times)) * 100
            ax5.text(width + 0.5, i, f'{count} ({percentage:.1f}%)', 
                    va='center', fontweight='bold')
        
        ax5.set_xlabel('Number of Queries', fontsize=12, fontweight='bold')
        ax5.set_title('Performance Distribution', fontsize=14, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
        
        # Main title
        fig.suptitle('Query Execution Performance Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        output_path = self.output_dir / "query_performance_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_all_visualizations(self):
        """Create all visualization charts"""
        if not self.metrics:
            print("⚠️  No metrics data available. Run some queries first!")
            return
        
        print("\n" + "="*70)
        print("🎨 GENERATING QUERY PERFORMANCE VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.create_response_time_histogram()
        self.create_response_time_timeline()
        self.create_entities_triples_chart()
        self.create_performance_summary_dashboard()
        
        print("\n" + "="*70)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*70)
        print(f"\n📁 Output directory: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*.png")):
            print(f"  • {file.name}")
        print()

def main():
    """Main execution function"""
    visualizer = QueryMetricsVisualizer()
    visualizer.create_all_visualizations()

if __name__ == '__main__':
    main()
