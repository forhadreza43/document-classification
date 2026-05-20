import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# ==========================================
# RVL-CDIP 16-CLASS LABEL MAPPING
# (Update if your label encoder uses different ordering)
# ==========================================
CLASS_NAMES = [
    'advertising', 'budget', 'email', 'form', 'invoice', 'letter',
    'memo', 'news_article', 'presentation', 'questionnaire', 'resume',
    'scientific_report', 'specification', 'technical_report', 'scientific_journal', 'handwritten'
]

# ==========================================
# FUNCTION: Compute Per-Class Accuracy
# ==========================================
def compute_per_class_accuracy(y_true, y_pred, num_classes=16):
    """Compute accuracy for each class individually"""
    per_class_acc = []
    for c in range(num_classes):
        # Mask for samples belonging to class c
        class_mask = (y_true == c)
        if np.sum(class_mask) == 0:
            per_class_acc.append(0.0)  # No samples for this class
            continue
        # Accuracy = correct predictions / total samples in class
        class_correct = np.sum((y_pred == c) & class_mask)
        class_total = np.sum(class_mask)
        acc = (class_correct / class_total) * 100
        per_class_acc.append(acc)
    return np.array(per_class_acc)

# ==========================================
# FUNCTION: Generate Class-wise Accuracy Bar Chart
# ==========================================
def plot_classwise_accuracy(save_dir=".", dpi=300, sort_by='improvement'):
    """
    Generate class-wise accuracy comparison bar chart
    
    Parameters:
    -----------
    save_dir : str
        Directory to save output figures
    dpi : int
        Resolution for PNG export
    sort_by : str
        How to order classes: 'improvement', 'alphabetical', or 'text_acc'
    """
    
    # 1. Load predictions
    y_true = np.load("eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_true.npy")
    y_pred_text = np.load("eval_outputs_text_50k/bert_margin_star_50k_y_pred_text.npy")
    y_pred_multi = np.load("eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_pred_multi.npy")
    
    assert len(y_true) == len(y_pred_text) == len(y_pred_multi), "❌ Length mismatch!"
    
    # 2. Compute per-class accuracies
    acc_text = compute_per_class_accuracy(y_true, y_pred_text)
    acc_multi = compute_per_class_accuracy(y_true, y_pred_multi)
    improvement = acc_multi - acc_text  # Positive = multimodal better
    
    # 3. Sort classes based on parameter
    if sort_by == 'improvement':
        # Sort by improvement (largest gain first)
        sort_idx = np.argsort(-improvement)  # Negative for descending
    elif sort_by == 'alphabetical':
        sort_idx = np.argsort(CLASS_NAMES)
    elif sort_by == 'text_acc':
        sort_idx = np.argsort(-acc_text)  # Text-only accuracy descending
    else:
        sort_idx = np.arange(len(CLASS_NAMES))  # Default order
    
    # Apply sorting
    sorted_names = [CLASS_NAMES[i] for i in sort_idx]
    sorted_acc_text = acc_text[sort_idx]
    sorted_acc_multi = acc_multi[sort_idx]
    sorted_improvement = improvement[sort_idx]
    
    # 4. Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Bar positions
    x = np.arange(len(sorted_names))
    bar_width = 0.35
    
    # Create bars
    bars_text = ax.bar(x - bar_width/2, sorted_acc_text, 
                       width=bar_width, label='Text-only (BERT)', 
                       color='#d62728', edgecolor='black', alpha=0.85, linewidth=0.8)
    
    bars_multi = ax.bar(x + bar_width/2, sorted_acc_multi, 
                        width=bar_width, label='Multimodal (LayoutLMv3)', 
                        color='#1f77b4', edgecolor='black', alpha=0.85, linewidth=0.8)
    
    # 5. Add value labels on bars (only for multimodal to reduce clutter)
    for bar, acc in zip(bars_multi, sorted_acc_multi):
        height = bar.get_height()
        # Only label if accuracy is reasonably high or improvement is significant
        if acc > 50 or (acc - sorted_acc_text[list(bars_multi).index(bar)]) > 10:
            ax.annotate(f'{acc:.1f}%', 
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color='#1f77b4')
    
    # 6. Add improvement annotations (arrows for significant gains)
    for i, (t_acc, m_acc, imp) in enumerate(zip(sorted_acc_text, sorted_acc_multi, sorted_improvement)):
        if imp > 15:  # Only annotate large improvements
            ax.annotate(f'+{imp:.1f}%', 
                        xy=(x[i], max(t_acc, m_acc)),
                        xytext=(0, 25),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#2ca02c'
                        )
    
    # 7. Formatting
    ax.set_xlabel('Document Category', fontsize=12, fontweight='500')
    ax.set_ylabel('Per-Class Accuracy (%)', fontsize=12, fontweight='500')
    # ax.set_title('Class-wise Accuracy Comparison: Multimodal vs. Text-only\n(50k Split, Test Set)', 
    #              fontsize=14, fontweight='600', pad=15)
    
    # X-axis labels (rotated for readability)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=10)
    
    # Y-axis
    ax.set_ylim(0, 115)  # Extra space for annotations
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='gray')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=False)
    
    # Add horizontal line at overall accuracy for reference
    overall_text = np.mean(sorted_acc_text)  # Approximate
    overall_multi = np.mean(sorted_acc_multi)
    ax.axhline(y=overall_multi, color='#1f77b4', linestyle=':', alpha=0.5, 
               label=f'Multi Overall: {overall_multi:.1f}%')
    
    # Tight layout
    plt.tight_layout()
    
    # 8. Save outputs
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_sorted_{sort_by}" if sort_by != 'improvement' else ""
    png_path = os.path.join(save_dir, f"classwise_accuracy_50k{suffix}.png")
    pdf_path = os.path.join(save_dir, f"classwise_accuracy_50k{suffix}.pdf")
    
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')  # Vector for thesis
    plt.close()
    
    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")
    
    # 9. Print summary statistics for report
    print("\n" + "="*70)
    print("Class-wise Accuracy Summary (50k Split)")
    print("="*70)
    print(f"{'Class':<20} {'Text-only':>10} {'Multimodal':>12} {'Improvement':>12}")
    print("-"*70)
    for name, t_acc, m_acc, imp in zip(sorted_names, sorted_acc_text, sorted_acc_multi, sorted_improvement):
        print(f"{name:<20} {t_acc:>9.1f}% {m_acc:>11.1f}% {imp:>+11.1f}%")
    print("-"*70)
    print(f"{'AVERAGE':<20} {np.mean(sorted_acc_text):>9.1f}% {np.mean(sorted_acc_multi):>11.1f}% {np.mean(sorted_improvement):>+11.1f}%")
    print("="*70 + "\n")
    
    return png_path, pdf_path

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Generate with different sorting options
    save_directory = "figures_classwise"
    
    # Primary: sorted by improvement (most impactful for thesis)
    plot_classwise_accuracy(save_dir=save_directory, dpi=300, sort_by='improvement')
    
    # Optional: also generate alphabetical version for appendix
    # plot_classwise_accuracy(save_dir=save_directory, dpi=300, sort_by='alphabetical')
    
    print("🎯 Figure ready for Section 4.1.2 (Result Tables and Graphs)")