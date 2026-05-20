import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# RVL-CDIP 16-class label mapping (must match text-only script)
CLASS_NAMES = [
    'advertising', 'budget', 'email', 'form', 'invoice', 'letter',
    'memo', 'news_article', 'presentation', 'questionnaire', 'resume',
    'scientific_report', 'specification', 'technical_report', 'scientific_journal', 'handwritten'
]

def plot_multimodal_cm(save_dir=".", dpi=300, normalize=False):
    # 1. Load predictions
    y_true = np.load("eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_true.npy")
    y_pred = np.load("eval_outputs_multimodal_50k/layoutlmv3_margin_star_50k_y_pred_multi.npy")
    
    assert len(y_true) == len(y_pred), "❌ Length mismatch!"
    
    # 2. Compute confusion matrix
    cm_kwargs = {'normalize': 'true'} if normalize else {}
    cm = confusion_matrix(y_true, y_pred, **cm_kwargs)
    fmt = '.2%' if normalize else 'd'
    title_suffix = "(Row-Normalized)" if normalize else "(Raw Counts)"
    
    # 3. Plot configuration
    plt.figure(figsize=(10, 8))
    cmap = "Greens"  # Distinct academic gradient for multimodal
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': 'Proportion' if normalize else 'Number of Documents'},
                linewidths=0.5, linecolor='lightgray')
    
    # plt.title(f"Confusion Matrix: Multimodal Model (LayoutLMv3)\n(50k Split, Test Set) {title_suffix}", 
    #           fontsize=14, fontweight='600', pad=12)
    plt.xlabel("Predicted Label", fontsize=12, fontweight='500')
    plt.ylabel("True Label", fontsize=12, fontweight='500')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # 4. Save outputs
    os.makedirs(save_dir, exist_ok=True)
    suffix = "_norm" if normalize else ""
    png_path = os.path.join(save_dir, f"cm_multimodal_50k{suffix}.png")
    pdf_path = os.path.join(save_dir, f"cm_multimodal_50k{suffix}.pdf")
    
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")

if __name__ == "__main__":
    # Run both raw and normalized versions
    plot_multimodal_cm(save_dir="figures_confusion", dpi=300, normalize=False)
    plot_multimodal_cm(save_dir="figures_confusion", dpi=300, normalize=True)