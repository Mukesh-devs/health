"""
Visualize Retrieval Performance Metrics
Generates comprehensive graphs from retrieval_metrics.log
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

# Set style for better-looking graphs
plt.style.use('seaborn-v0_8-darkgrid')

class RetrievalMetricsVisualizer:
    """Visualize retrieval performance metrics"""
    
    def __init__(self):
        self.output_dir = Path("performance/graphs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance data from retrieval_metrics.log
        self.overall_metrics = {
            'Precision': 0.7500,
            'Recall': 0.5510,
            'F1 Score': 0.6353,
            'MRR': 0.7576,
            'MAP': 0.6364
        }
        
        self.precision_at_k = {
            'P@1': 0.7576,
            'P@3': 0.6869,
            'P@5': 0.6869,
            'P@10': 0.6869
        }
        
        self.recall_at_k = {
            'R@1': 0.6061,
            'R@3': 0.6364,
            'R@5': 0.6364,
            'R@10': 0.6364
        }
        
        self.f1_at_k = {
            'F1@1': 0.6566,
            'F1@3': 0.6354,
            'F1@5': 0.6354,
            'F1@10': 0.6354
        }
    
    def create_overall_metrics_bar_chart(self):
        """Create bar chart for overall retrieval metrics"""
        print("📊 Creating overall metrics bar chart...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metrics = list(self.overall_metrics.keys())
        values = list(self.overall_metrics.values())
        
        # Create bars with gradient colors
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.4f}\n({value*100:.2f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Styling
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
        ax.set_title('Overall Retrieval Performance Metrics\n(ADAPTIVE_PRECISION Strategy)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add horizontal reference lines
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='70% Target')
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / "overall_metrics_bar_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_precision_recall_f1_comparison(self):
        """Create grouped bar chart comparing P, R, F1 at different K values"""
        print("📊 Creating Precision-Recall-F1 comparison chart...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Data
        k_values = ['K=1', 'K=3', 'K=5', 'K=10']
        precision = [0.7576, 0.6869, 0.6869, 0.6869]
        recall = [0.6061, 0.6364, 0.6364, 0.6364]
        f1 = [0.6566, 0.6354, 0.6354, 0.6354]
        
        x = np.arange(len(k_values))
        width = 0.25
        
        # Create bars
        bars1 = ax.bar(x - width, precision, width, label='Precision', 
                      color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, recall, width, label='Recall', 
                      color='#2ecc71', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, f1, width, label='F1 Score', 
                      color='#9b59b6', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Styling
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_xlabel('Top-K Results', fontsize=14, fontweight='bold')
        ax.set_title('Precision, Recall, and F1 Score @ K\n(Entity Retrieval Performance)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(k_values)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        output_path = self.output_dir / "precision_recall_f1_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_metrics_at_k_line_plot(self):
        """Create line plot showing how metrics change with K"""
        print("📊 Creating metrics @ K line plot...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        k_values = [1, 3, 5, 10]
        precision = [0.7576, 0.6869, 0.6869, 0.6869]
        recall = [0.6061, 0.6364, 0.6364, 0.6364]
        f1 = [0.6566, 0.6354, 0.6354, 0.6354]
        
        # Plot lines
        ax.plot(k_values, precision, marker='o', linewidth=3, markersize=10, 
               label='Precision@K', color='#3498db')
        ax.plot(k_values, recall, marker='s', linewidth=3, markersize=10, 
               label='Recall@K', color='#2ecc71')
        ax.plot(k_values, f1, marker='^', linewidth=3, markersize=10, 
               label='F1@K', color='#9b59b6')
        
        # Add value annotations
        for i, k in enumerate(k_values):
            ax.annotate(f'{precision[i]:.4f}', (k, precision[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            ax.annotate(f'{recall[i]:.4f}', (k, recall[i]), 
                       textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
            ax.annotate(f'{f1[i]:.4f}', (k, f1[i]), 
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        # Styling
        ax.set_xlabel('K (Top-K Results)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Retrieval Metrics vs K Value\n(Performance at Different Retrieval Depths)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(k_values)
        ax.set_ylim(0.5, 0.85)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=12, framealpha=0.9)
        
        plt.tight_layout()
        output_path = self.output_dir / "metrics_at_k_line_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_performance_radar_chart(self):
        """Create radar chart for overall performance visualization"""
        print("📊 Creating performance radar chart...")
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Data
        metrics = ['Precision', 'Recall', 'F1 Score', 'MRR', 'MAP']
        values = [0.7500, 0.5510, 0.6353, 0.7576, 0.6364]
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=3, color='#3498db', label='Performance')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        
        # Fix axis to go in the right order
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        
        # Add value labels
        for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
            ax.text(angle, value + 0.05, f'{value:.4f}', 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_title('Overall Retrieval Performance Radar\n(ADAPTIVE_PRECISION Strategy)', 
                    fontsize=16, fontweight='bold', pad=30)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        output_path = self.output_dir / "performance_radar_chart.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with multiple subplots"""
        print("📊 Creating comprehensive performance dashboard...")
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall Metrics Bar Chart (top left, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        metrics = list(self.overall_metrics.keys())
        values = list(self.overall_metrics.values())
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.set_title('Overall Metrics', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_axisbelow(True)
        
        # 2. Precision vs Recall scatter (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.scatter([0.7500], [0.5510], s=500, c='#e74c3c', alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_xlabel('Precision', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Recall', fontsize=11, fontweight='bold')
        ax2.set_title('P-R Trade-off', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.annotate(f'P={0.7500:.3f}\nR={0.5510:.3f}', 
                    xy=(0.7500, 0.5510), xytext=(0.6, 0.7),
                    arrowprops=dict(arrowstyle='->', lw=2), fontsize=10, fontweight='bold')
        
        # 3. Precision @ K (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        k_vals = [1, 3, 5, 10]
        p_vals = [0.7576, 0.6869, 0.6869, 0.6869]
        ax3.plot(k_vals, p_vals, marker='o', linewidth=3, markersize=10, color='#3498db')
        ax3.fill_between(k_vals, p_vals, alpha=0.3, color='#3498db')
        ax3.set_xlabel('K', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Precision', fontsize=11, fontweight='bold')
        ax3.set_title('Precision @ K', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0.5, 0.85)
        
        # 4. Recall @ K (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        r_vals = [0.6061, 0.6364, 0.6364, 0.6364]
        ax4.plot(k_vals, r_vals, marker='s', linewidth=3, markersize=10, color='#2ecc71')
        ax4.fill_between(k_vals, r_vals, alpha=0.3, color='#2ecc71')
        ax4.set_xlabel('K', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Recall', fontsize=11, fontweight='bold')
        ax4.set_title('Recall @ K', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0.5, 0.85)
        
        # 5. F1 @ K (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        f1_vals = [0.6566, 0.6354, 0.6354, 0.6354]
        ax5.plot(k_vals, f1_vals, marker='^', linewidth=3, markersize=10, color='#9b59b6')
        ax5.fill_between(k_vals, f1_vals, alpha=0.3, color='#9b59b6')
        ax5.set_xlabel('K', fontsize=11, fontweight='bold')
        ax5.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
        ax5.set_title('F1 @ K', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0.5, 0.85)
        
        # 6. MRR and MAP comparison (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        ranking_metrics = ['MRR', 'MAP']
        ranking_values = [0.7576, 0.6364]
        bars = ax6.bar(ranking_metrics, ranking_values, color=['#e74c3c', '#f39c12'], 
                      alpha=0.8, edgecolor='black')
        for bar, value in zip(bars, ranking_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax6.set_ylim(0, 1.0)
        ax6.set_title('Ranking Metrics', fontsize=14, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        ax6.set_axisbelow(True)
        
        # 7. Performance rating gauge (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        f1_score = 0.6353
        categories = ['Poor\n(<0.4)', 'Fair\n(0.4-0.6)', 'Good\n(0.6-0.75)', 'Excellent\n(>0.75)']
        colors_gauge = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
        y_pos = np.arange(len(categories))
        bars = ax7.barh(y_pos, [0.4, 0.2, 0.15, 0.25], color=colors_gauge, alpha=0.3)
        
        # Highlight current performance
        if f1_score < 0.4:
            highlight_idx = 0
        elif f1_score < 0.6:
            highlight_idx = 1
        elif f1_score < 0.75:
            highlight_idx = 2
        else:
            highlight_idx = 3
        
        bars[highlight_idx].set_alpha(0.9)
        bars[highlight_idx].set_edgecolor('black')
        bars[highlight_idx].set_linewidth(3)
        
        ax7.set_yticks(y_pos)
        ax7.set_yticklabels(categories)
        ax7.set_xlabel('Range', fontsize=11, fontweight='bold')
        ax7.set_title(f'Performance Rating\nF1={f1_score:.4f}', fontsize=14, fontweight='bold')
        ax7.set_xlim(0, 1)
        
        # 8. Strategy info (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        info_text = (
            "RETRIEVAL STRATEGY\n"
            "─────────────────────\n\n"
            "Strategy: ADAPTIVE_PRECISION\n"
            "Threshold: 0.88\n"
            "Top-K: 10\n"
            "Soft Matching: Enabled\n\n"
            "DATASET INFO\n"
            "─────────────────────\n\n"
            "Queries: 33\n"
            "Entities: 162,212\n"
            "Model: MiniLM-L6-v2\n\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        ax8.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Main title
        fig.suptitle('Entity Retrieval Performance Dashboard - Comprehensive Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        output_path = self.output_dir / "comprehensive_dashboard.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_all_visualizations(self):
        """Create all visualization charts"""
        print("\n" + "="*70)
        print("🎨 GENERATING RETRIEVAL PERFORMANCE VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.create_overall_metrics_bar_chart()
        self.create_precision_recall_f1_comparison()
        self.create_metrics_at_k_line_plot()
        self.create_performance_radar_chart()
        self.create_comprehensive_dashboard()
        
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
    visualizer = RetrievalMetricsVisualizer()
    visualizer.create_all_visualizations()

if __name__ == '__main__':
    main()
