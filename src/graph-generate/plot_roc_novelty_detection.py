import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc

# ==========================================
# ACADEMIC COLOR PALETTE
# ==========================================
PALETTE = {
    'text_only': '#C0392B',      # Crimson
    'multimodal': '#1B4F72',     # Navy
    'knn': '#27AE60',            # Green for KNN
    'knn_star': '#2E86AB',       # Blue for KNN*
    'grid': '#EAECEE',
    'text': '#2C3E50',
    'bg': '#FFFFFF'
}

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': '600',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
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
# FUNCTION: Load Novelty Scores and Compute ROC
# ==========================================
def compute_novelty_roc(id_scores_path, ood_scores_path):
    """Compute ROC curve for novelty detection (ID vs OOD)"""
    id_scores = np.load(id_scores_path)  # Negative distances (higher = more ID-like)
    ood_scores = np.load(ood_scores_path)
    
    # Create labels: 1 = ID, 0 = OOD
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, thresholds

# ==========================================
# FUNCTION: Plot Novelty Detection ROC Comparison
# ==========================================
def plot_novelty_roc(save_dir=".", dpi=300):
    """Generate ROC curves for novelty detection: KNN vs KNN*, Text vs Multimodal"""
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # === TEXT-ONLY MODEL ===
    # KNN
    fpr_text_knn, tpr_text_knn, auc_text_knn, _ = compute_novelty_roc(
        "novelty_outputs_text/bert_margin_star_id_scores_knn.npy",
        "novelty_outputs_text/bert_margin_star_ood_scores_knn.npy"
    )
    ax.plot(fpr_text_knn, tpr_text_knn, 
            color=PALETTE['text_only'], linestyle='-', linewidth=2, alpha=0.8,
            label=f'Text-only + KNN — AUC = {auc_text_knn:.3f}')
    
    # KNN*
    fpr_text_knnstar, tpr_text_knnstar, auc_text_knnstar, _ = compute_novelty_roc(
        "novelty_outputs_text/bert_margin_star_id_scores_knn_star.npy",
        "novelty_outputs_text/bert_margin_star_ood_scores_knn_star.npy"
    )
    ax.plot(fpr_text_knnstar, tpr_text_knnstar, 
            color=PALETTE['text_only'], linestyle='--', linewidth=2,
            label=f'Text-only + KNN* — AUC = {auc_text_knnstar:.3f}')
    
    # === MULTIMODAL MODEL ===
    # KNN
    fpr_multi_knn, tpr_multi_knn, auc_multi_knn, _ = compute_novelty_roc(
        "novelty_outputs_multimodal/layoutlmv3_margin_star_id_scores_knn.npy",
        "novelty_outputs_multimodal/layoutlmv3_margin_star_ood_scores_knn.npy"
    )
    ax.plot(fpr_multi_knn, tpr_multi_knn, 
            color=PALETTE['multimodal'], linestyle='-', linewidth=2.5,
            label=f'Multimodal + KNN — AUC = {auc_multi_knn:.3f}')
    
    # KNN*
    fpr_multi_knnstar, tpr_multi_knnstar, auc_multi_knnstar, _ = compute_novelty_roc(
        "novelty_outputs_multimodal/layoutlmv3_margin_star_id_scores_knn_star.npy",
        "novelty_outputs_multimodal/layoutlmv3_margin_star_ood_scores_knn_star.npy"
    )
    ax.plot(fpr_multi_knnstar, tpr_multi_knnstar, 
            color=PALETTE['multimodal'], linestyle='--', linewidth=2.5,
            label=f'Multimodal + KNN* — AUC = {auc_multi_knnstar:.3f}')
    
    # Diagonal (random)
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='500')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='500')
    # ax.set_title('ROC Curve: Novelty Detection (ID vs OOD)\n(RVL-CDIP + RVL-CDIP-O, 50k Training)', 
    #              fontsize=13, fontweight='600', pad=15)
    
    ax.legend(loc='lower right', fontsize=9, frameon=True, ncol=2)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    
    # Add FPR@TPR95 annotation (from your results)
    fpr95_multi_knn = 46.96  # From novelty_detection_FRP95.txt
    fpr95_text_knn = 54.65
    ax.text(0.55, 0.12, f'FPR@TPR95:\nMulti+KNN: {fpr95_multi_knn:.1f}%\nText+KNN: {fpr95_text_knn:.1f}%', 
            fontsize=9, fontweight='bold', color=PALETTE['multimodal'],
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor=PALETTE['multimodal'], alpha=0.85))
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'roc_novelty_detection_50k.png'), dpi=dpi, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.savefig(os.path.join(save_dir, 'roc_novelty_detection_50k.pdf'), bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    
    print(f"✅ Novelty ROC curves saved to '{save_dir}/'")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    save_directory = "figures_roc_pr"
    
    print("\n" + "="*60)
    print("Generating Novelty Detection ROC Curves")
    print("="*60 + "\n")
    
    plot_novelty_roc(save_dir=save_directory, dpi=300)
    
    print("🎯 Figure ready for Section 4.1.2 / 4.4.2 of your thesis!")