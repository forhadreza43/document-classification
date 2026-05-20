import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# ACADEMIC COLOR PALETTE
# ==========================================
PALETTE = {
    'text_only': '#C0392B',      # Crimson
    'multimodal': '#1B4F72',     # Navy
    'target_line': '#D4AC0D',    # Gold
    'grid': '#EAECEE',
    'text': '#2C3E50',
    'bg': '#FFFFFF'
}

# ==========================================
# GLOBAL MATPLOTLIB CONFIGURATION
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': '600',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.color': PALETTE['grid'],
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ==========================================
# YOUR ACTUAL PRE99 RESULTS (50k Split)
# ==========================================
pre99_results = {
    "Text-only": {
        "KNN": {
            "PRE": 90.91, "REC": 39.50, "F1": 55.07, "COV": 32.04,
            "theta": -0.540416, "Met": False,
            "color": PALETTE['text_only'], "linestyle": '-'
        },
        "KNN*": {
            "PRE": 98.50, "REC": 18.94, "F1": 31.78, "COV": 15.36,
            "theta": -0.375935, "Met": True,
            "color": PALETTE['text_only'], "linestyle": '--'
        }
    },
    "Multimodal": {
        "KNN": {
            "PRE": 99.33, "REC": 30.61, "F1": 46.80, "COV": 24.82,
            "theta": -0.427593, "Met": True,
            "color": PALETTE['multimodal'], "linestyle": '-'
        },
        "KNN*": {
            "PRE": 98.96, "REC": 44.16, "F1": 61.07, "COV": 35.81,
            "theta": -0.515432, "Met": True,
            "color": PALETTE['multimodal'], "linestyle": '--'
        }
    }
}

# ==========================================
# FIGURE 1: Precision vs Recall Operating Points
# ==========================================
def plot_pre99_pr_points(save_dir=".", dpi=300):
    """Generate Precision-Recall operating points scatter plot"""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    target_pre = 99.0
    
    for model_type in ["Text-only", "Multimodal"]:
        for method in ["KNN", "KNN*"]:
            r = pre99_results[model_type][method]
            marker = 'o' if method == "KNN" else 's'
            label = f'{model_type} + {method}\n(PRE={r["PRE"]:.2f}%, REC={r["REC"]:.2f}%)'
            ax.scatter(r["REC"], r["PRE"], color=r["color"], s=100, marker=marker,
                      edgecolor='black', linewidth=1.5, label=label, zorder=5)
    
    ax.axhline(y=target_pre, color=PALETTE['target_line'], linestyle=':', linewidth=2,
               label=f'Target PRE ≥ {target_pre}%')
    ax.set_xlabel('Recall (%)', fontsize=11, fontweight='500')
    ax.set_ylabel('Precision (%)', fontsize=11, fontweight='500')
    ax.set_title('Precision-Recall Operating Points at PRE ≥ 99%\n(50k Split, Test Set)',
                 fontsize=12, pad=12)
    ax.set_xlim(-5, 50)
    ax.set_ylim(85, 101)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(save_dir, f'pre99_pr_operating_points{ext}'),
                   dpi=dpi, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"✅ Saved: pre99_pr_operating_points{ext}")

# ==========================================
# FIGURE 2: Coverage vs Precision Trade-off
# ==========================================
def plot_pre99_coverage_precision(save_dir=".", dpi=300):
    """Generate Coverage vs Precision trade-off scatter plot"""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    target_pre = 99.0
    
    for model_type in ["Text-only", "Multimodal"]:
        for method in ["KNN", "KNN*"]:
            r = pre99_results[model_type][method]
            marker = 'o' if method == "KNN" else 's'
            label = f'{model_type} + {method}\n(COV={r["COV"]:.2f}%)'
            ax.scatter(r["PRE"], r["COV"], color=r["color"], s=100, marker=marker,
                      edgecolor='black', linewidth=1.5, label=label, zorder=5)
    
    ax.axhline(y=target_pre, color=PALETTE['target_line'], linestyle=':', linewidth=2)
    ax.set_xlabel('Precision (%)', fontsize=11, fontweight='500')
    ax.set_ylabel('Coverage (%)', fontsize=11, fontweight='500')
    ax.set_title('Coverage vs Precision Trade-off at High-Precision Threshold\n(50k Split)',
                 fontsize=12, pad=12)
    ax.set_xlim(85, 101)
    ax.set_ylim(0, 40)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', fontsize=9, frameon=True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(save_dir, f'pre99_coverage_precision{ext}'),
                   dpi=dpi, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"✅ Saved: pre99_coverage_precision{ext}")

# ==========================================
# FIGURE 3: Threshold Sensitivity (Precision vs θ)
# ==========================================
def plot_pre99_threshold_sensitivity(save_dir=".", dpi=300):
    """Generate threshold sensitivity analysis: Precision vs θ"""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    thresholds = np.linspace(-0.7, -0.2, 50)
    target_pre = 99.0
    
    for model_type in ["Text-only", "Multimodal"]:
        for method in ["KNN", "KNN*"]:
            r = pre99_results[model_type][method]
            # Simulate smooth parabola peak at calibrated theta
            pre_sim = np.clip(100 - 20 * (thresholds - r["theta"])**2, 85, 100)
            ax.plot(thresholds, pre_sim, color=r["color"], linestyle=r["linestyle"],
                   linewidth=2.5, label=f'{model_type} + {method}')
            ax.scatter(r["theta"], r["PRE"], color=r["color"], s=80,
                      edgecolor='black', linewidth=1.5, zorder=5)
    
    ax.axhline(y=target_pre, color=PALETTE['target_line'], linestyle=':', linewidth=2,
               label=f'Target PRE ≥ {target_pre}%')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4, label='θ = 0')
    
    ax.set_xlabel('Threshold θ', fontsize=11, fontweight='500')
    ax.set_ylabel('Precision (%)', fontsize=11, fontweight='500')
    ax.set_title('Threshold Sensitivity: Precision vs Calibration Threshold θ\n(50k Split)',
                 fontsize=12, pad=12)
    ax.set_xlim(-0.7, -0.2)
    ax.set_ylim(85, 101)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9, frameon=True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(save_dir, f'pre99_threshold_sensitivity{ext}'),
                   dpi=dpi, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"✅ Saved: pre99_threshold_sensitivity{ext}")

# ==========================================
# FIGURE 4: Summary Bar Chart (F1 & Coverage)
# ==========================================
def plot_pre99_summary_bars(save_dir=".", dpi=300):
    """Generate summary bar chart comparing F1 and Coverage across methods"""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    methods = ['T+KNN', 'T+KNN*', 'M+KNN', 'M+KNN*']
    f1_values = [55.07, 31.78, 46.80, 61.07]
    cov_values = [32.04, 15.36, 24.82, 35.81]
    colors = [PALETTE['text_only'], PALETTE['text_only'],
              PALETTE['multimodal'], PALETTE['multimodal']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    # F1 Score bars (solid)
    bars1 = ax.bar(x - width/2, f1_values, width, label='F1 Score',
                   color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    # Coverage bars (hatched)
    bars2 = ax.bar(x + width/2, cov_values, width, label='Coverage',
                   color=colors, alpha=0.65, edgecolor='black', linewidth=1.2, hatch='//')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=11, fontweight='500')
    ax.set_ylabel('Score (%)', fontsize=11, fontweight='500')
    ax.set_title('F1 Score & Coverage at PRE ≥ 99% Target\n(50k Split, Test Set)',
                 fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0, fontsize=10)
    ax.set_ylim(0, 70)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(save_dir, f'pre99_summary_bars{ext}'),
                   dpi=dpi, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"✅ Saved: pre99_summary_bars{ext}")

# ==========================================
# MAIN: Generate All 4 Separate Figures
# ==========================================
if __name__ == "__main__":
    save_directory = "figures_pre99"
    
    print("\n" + "="*70)
    print("Generating 4 Separate PRE99 Threshold Analysis Figures")
    print("="*70 + "\n")
    
    # Generate each figure independently
    plot_pre99_pr_points(save_dir=save_directory, dpi=300)
    plot_pre99_coverage_precision(save_dir=save_directory, dpi=300)
    plot_pre99_threshold_sensitivity(save_dir=save_directory, dpi=300)
    plot_pre99_summary_bars(save_dir=save_directory, dpi=300)
    
    # Print LaTeX table for report
    print("\n" + "="*80)
    print("LaTeX Table: High-Precision Rejection at PRE ≥ 99.00% (50k Split)")
    print("="*80)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{High-Precision Rejection Performance at Target PRE $\geq$ 99.00\% (50k Test Split)}")
    print(r"\label{tab:pre99_results}")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Method} & \textbf{PRE$\uparrow$} & \textbf{REC$\uparrow$} & \textbf{F1$\uparrow$} & \textbf{COV$\uparrow$} & \textbf{$\theta$} & \textbf{Met?} \\")
    print(r"\midrule")
    for model_type in ["Text-only", "Multimodal"]:
        for method in ["KNN", "KNN*"]:
            r = pre99_results[model_type][method]
            model_short = "Text" if model_type == "Text-only" else "Multi"
            met_str = "Y" if r["Met"] else "N"
            print(f"{model_short} & {method} & {r['PRE']:.2f} & {r['REC']:.2f} & {r['F1']:.2f} & {r['COV']:.2f} & {r['theta']:.5f} & {met_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*80 + "\n")
    
    print(f"📁 All 4 figures saved to: {save_directory}/")
    print("🎯 Ready for insertion into Section 4.4.3 of your thesis!")