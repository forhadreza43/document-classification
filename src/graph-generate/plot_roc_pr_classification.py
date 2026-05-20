import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import os

# ==========================================
# ACADEMIC COLOR PALETTE
# ==========================================
PALETTE = {
    'text_only': '#C0392B',      # Crimson for text-only
    'multimodal': '#1B4F72',     # Navy for multimodal
    'grid': '#EAECEE',
    'text': '#2C3E50',
    'bg': '#FFFFFF'
}

# RVL-CDIP 16-class label mapping
CLASS_NAMES = [
    'advertising', 'budget', 'email', 'form', 'invoice', 'letter',
    'memo', 'news_article', 'presentation', 'questionnaire', 'resume',
    'scientific_report', 'specification', 'technical_report', 'scientific_journal', 'handwritten'
]
NUM_CLASSES = len(CLASS_NAMES)

# ==========================================
# GLOBAL MATPLOTLIB CONFIGURATION
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': '600',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 16,
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
# FUNCTION: Load and Compute Curves
# ==========================================
def compute_roc_pr_curves(y_true_path, y_probs_path, num_classes=16):
    """Load predictions and compute ROC/PR curves (one-vs-rest)"""
    y_true = np.load(y_true_path)
    y_probs = np.load(y_probs_path)
    
    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    
    # Compute ROC and PR for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    
    # Micro-average (aggregate all classes)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_probs.ravel())
    avg_precision["micro"] = average_precision_score(y_true_bin.ravel(), y_probs.ravel())
    
    return fpr, tpr, roc_auc, precision, recall, avg_precision

# ==========================================
# FUNCTION: Plot ROC Curves Comparison
# ==========================================
def plot_roc_comparison(save_dir=".", dpi=300):
    """Generate ROC curve comparison: Text-only vs Multimodal"""
    
    # Load curves for both models
    fpr_text, tpr_text, roc_auc_text, _, _, _ = compute_roc_pr_curves(
        "eval_outputs_text_50k/bert_margin_star_50k_y_true.npy",
        "eval_outputs_text/bert_margin_star_y_probs_text.npy"
    )
    
    fpr_multi, tpr_multi, roc_auc_multi, _, _, _ = compute_roc_pr_curves(
        "eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_true.npy",
        "eval_outputs_multimodal/layoutlmv3_margin_star_y_probs_multi.npy"
    )
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot micro-average ROC curves
    ax.plot(fpr_text["micro"], tpr_text["micro"], 
            color=PALETTE['text_only'], linestyle='-', linewidth=2.5,
            label=f'Text-only (BERT) — AUC = {roc_auc_text["micro"]:.3f}')
    
    ax.plot(fpr_multi["micro"], tpr_multi["micro"], 
            color=PALETTE['multimodal'], linestyle='-', linewidth=2.5,
            label=f'Multimodal (LayoutLMv3) — AUC = {roc_auc_multi["micro"]:.3f}')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=16, fontweight='500')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=16, fontweight='500')
    # ax.set_title('ROC Curve: Multi-Class Classification (One-vs-Rest)\n(50k Split, Test Set, Micro-Average)', 
    #              fontsize=13, fontweight='600', pad=15)
    
    ax.legend(loc='lower right', fontsize=16, frameon=True)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    
    # Add AUC gain annotation
    auc_gain = roc_auc_multi["micro"] - roc_auc_text["micro"]
    ax.text(0.65, 0.15, f'AUC Gain: +{auc_gain:.3f}', 
            fontsize=16, fontweight='bold', color=PALETTE['multimodal'],
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor=PALETTE['multimodal'], alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'roc_classification_50k.png'), dpi=dpi, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.savefig(os.path.join(save_dir, 'roc_classification_50k.pdf'), bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    
    print(f"✅ ROC curves saved to '{save_dir}/'")

# ==========================================
# FUNCTION: Plot PR Curves Comparison
# ==========================================
def plot_pr_comparison(save_dir=".", dpi=300):
    """Generate PR curve comparison: Text-only vs Multimodal"""
    
    # Load curves for both models
    _, _, _, precision_text, recall_text, ap_text = compute_roc_pr_curves(
        "eval_outputs_text_50k/bert_margin_star_50k_y_true.npy",
        "eval_outputs_text/bert_margin_star_y_probs_text.npy"
    )
    
    _, _, _, precision_multi, recall_multi, ap_multi = compute_roc_pr_curves(
        "eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_true.npy",
        "eval_outputs_multimodal/layoutlmv3_margin_star_y_probs_multi.npy"
    )
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot micro-average PR curves
    ax.plot(recall_text["micro"], precision_text["micro"], 
            color=PALETTE['text_only'], linestyle='-', linewidth=2.5,
            label=f'Text-only (BERT) — AP = {ap_text["micro"]:.3f}')
    
    ax.plot(recall_multi["micro"], precision_multi["micro"], 
            color=PALETTE['multimodal'], linestyle='-', linewidth=2.5,
            label=f'Multimodal (LayoutLMv3) — AP = {ap_multi["micro"]:.3f}')
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=16, fontweight='500')
    ax.set_ylabel('Precision', fontsize=16, fontweight='500')
    # ax.set_title('Precision-Recall Curve: Multi-Class Classification\n(50k Split, Test Set, Micro-Average)', 
    #              fontsize=13, fontweight='600', pad=15)
    
    ax.legend(loc='lower left', fontsize=16, frameon=True)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    
    # Add AP gain annotation
    ap_gain = ap_multi["micro"] - ap_text["micro"]
    ax.text(0.2, 0.95, f'AP Gain: +{ap_gain:.3f}', 
            fontsize=16, fontweight='bold', color=PALETTE['multimodal'],
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor=PALETTE['multimodal'], alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'pr_classification_50k.png'), dpi=dpi, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.savefig(os.path.join(save_dir, 'pr_classification_50k.pdf'), bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    
    print(f"✅ PR curves saved to '{save_dir}/'")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    save_directory = "figures_roc_pr"
    
    print("\n" + "="*60)
    print("Generating Classification ROC/PR Curves")
    print("="*60 + "\n")
    
    plot_roc_comparison(save_dir=save_directory, dpi=300)
    plot_pr_comparison(save_dir=save_directory, dpi=300)
    
    print("🎯 Figures ready for Section 4.1.2 of your thesis!")