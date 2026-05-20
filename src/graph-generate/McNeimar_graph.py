import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# ==========================================
# YOUR ACTUAL McNEMAR RESULTS (All Splits)
# ==========================================
results = {
    "10k": {
        "n01": 864,   # Text wrong, Multimodal correct ✅
        "n10": 20,    # Text correct, Multimodal wrong ❌
        "chi2": 803.90,
        "odds_ratio": 43.20,
        "test_size": 1250
    },
    "30k": {
        "n01": 2882,
        "n10": 32,
        "chi2": 2785.45,
        "odds_ratio": 90.06,
        "test_size": 3750
    },
    "50k": {
        "n01": 4966,
        "n10": 53,
        "chi2": 4807.28,
        "odds_ratio": 93.70,
        "test_size": 6250
    }
}

# ==========================================
# FUNCTION: Generate Single McNemar Chart
# ==========================================
def plot_mcnemar_single(split_name, n01, n10, chi2, odds_ratio, test_size, save_dir="."):
    """Generate McNemar discordant pairs bar chart for a single split"""
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    # Categories and values
    categories = ['Multimodal corrects\ntext errors (n₀₁)', 
                  'Text corrects\nmultimodal errors (n₁₀)']
    values = [n01, n10]
    colors = ['#2ecc71', '#e74c3c']  # Green and Red
    
    # Create bars
    bars = ax.bar(categories, values, color=colors, edgecolor='black', 
                  linewidth=1.5, width=0.6, alpha=0.95)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{value:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11,
                    )
    
    # Add statistics box (top right)
    p_str = "< 10⁻⁵⁰"
    stats_text = (f'χ² = {chi2:,.1f}\n'
                  f'p {p_str}\n'
                  f'Odds Ratio = {odds_ratio:.1f}×')
    
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', 
                      edgecolor='#6c757d', alpha=0.95, linewidth=1.5))
    
    # Formatting
    ax.set_ylabel('Number of Documents', fontsize=12, fontweight='500', labelpad=10)
    ax.set_xlabel('Discordant Pairs', fontsize=12, fontweight='500', labelpad=10)


    # Y-axis scaling
    ax.set_ylim(0, max(values) * 1.2)
    
    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # X-axis labels
    plt.xticks(rotation=0, ha='center', fontsize=11)
    
    # Add annotation arrow highlighting the gap
    if n01 > n10 * 2:  # Only add if gap is significant
        ax.annotate('', 
                    xy=(0, n01*0.5), xytext=(1, n01*0.5),
                    arrowprops=dict(arrowstyle='<->', color='blue', lw=2, ls='--', alpha=0.7))
        ax.text(0.5, n01*0.52, f'{odds_ratio:.1f}× difference', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='blue', alpha=0.9)
    
    # Save and show
    plt.tight_layout()
    output_path = os.path.join(save_dir, f'mcnemar_discordant_{split_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    return output_path


# ==========================================
# FUNCTION: Generate Combined Comparison Chart
# ==========================================
def plot_mcnemar_combined(results, save_dir="."):
    """Generate side-by-side McNemar charts for all splits"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("McNemar's Discordant Pairs: Multimodal vs. Text-only\n(All Training Splits)", 
                 fontsize=16, fontweight='600', y=1.02)
    
    colors = ['#2ecc71', '#e74c3c']
    split_labels = ['10k', '30k', '50k']
    
    for idx, (split_name, data) in enumerate(zip(split_labels, [results["10k"], results["30k"], results["50k"]])):
        ax = axes[idx]
        
        categories = ['n₀₁\n(Multi corrects\nText)', 'n₁₀\n(Text corrects\nMulti)']
        values = [data["n01"], data["n10"]]
        
        # Create bars
        bars = ax.bar(categories, values, color=colors, edgecolor='black', 
                      linewidth=1.2, width=0.6, alpha=0.95)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{value:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        
        # Stats box
        stats_text = (f'χ²={data["chi2"]:.0f}\n'
                      f'OR={data["odds_ratio"]:.1f}×')
        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))
        
        # Formatting
        ax.set_ylabel('Documents' if idx == 0 else '', fontsize=10)
        ax.set_title(f'{split_name} Split\n(n={data["test_size"]:,})', fontsize=11, fontweight='500')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(0, max(values) * 1.25)
        ax.tick_params(axis='x', labelsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(save_dir, 'mcnemar_all_splits_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved combined chart: {output_path}")
    return output_path


# ==========================================
# MAIN: Generate All Charts
# ==========================================
if __name__ == "__main__":
    save_directory = "figures_mcnemar"
    os.makedirs(save_directory, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating McNemar Discordant Pairs Charts")
    print("="*60 + "\n")
    
    # Generate individual charts
    for split_name, data in results.items():
        plot_mcnemar_single(
            split_name=split_name,
            n01=data["n01"],
            n10=data["n10"],
            chi2=data["chi2"],
            odds_ratio=data["odds_ratio"],
            test_size=data["test_size"],
            save_dir=save_directory
        )
    
    # Generate combined comparison chart
    plot_mcnemar_combined(results, save_dir=save_directory)
    
    # Print summary table for report
    print("\n" + "="*60)
    print("McNemar Test Summary (For Thesis Table)")
    print("="*60)
    print(f"{'Split':<8} {'n₀₁':>8} {'n₁₀':>8} {'χ²':>12} {'Odds Ratio':>12} {'p-value':>10}")
    print("-"*60)
    for split_name, data in results.items():
        print(f"{split_name:<8} {data['n01']:>8,} {data['n10']:>8,} "
              f"{data['chi2']:>12.2f} {data['odds_ratio']:>12.2f} {'< 10⁻⁵⁰':>10}")
    print("="*60 + "\n")
    
    print("📁 All figures saved to:", save_directory)
    print("🎯 Ready for insertion into Section 3.4.4 and 4.1.3 of your report!")