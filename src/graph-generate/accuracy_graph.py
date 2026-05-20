import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. YOUR ACTUAL RESULTS (from provided files)
# ==========================================
x_labels = ['10k', '30k', '50k']
training_samples = np.array([10000, 30000, 50000])

# Test set accuracies
acc_multi = np.array([77.52, 87.84, 88.78])  # Multimodal
acc_text = np.array([23.04, 72.45, 77.94])   # Text-only (BERT)

# Corresponding test set sizes (from your project structure)
n_test = np.array([1250, 3750, 6250])

# ==========================================
# 2. COMPUTE 95% CONFIDENCE INTERVALS
# Formula: CI = 1.96 * sqrt(p*(1-p)/n) * 100
# ==========================================
def compute_ci(acc_pct, n):
    p = acc_pct / 100.0
    se = np.sqrt(p * (1 - p) / n)
    return 1.96 * se * 100  # Convert back to percentage

ci_multi = compute_ci(acc_multi, n_test)
ci_text = compute_ci(acc_text, n_test)

# ==========================================
# 3. PLOT CONFIGURATION
# ==========================================
plt.figure(figsize=(8.5, 6))
plt.grid(True, linestyle=':', alpha=0.7, color='gray')

# Multimodal line
plt.plot(training_samples, acc_multi, marker='o', linestyle='-', 
         color='#1f77b4', linewidth=2.5, markersize=8, label='Multimodal')
plt.errorbar(training_samples, acc_multi, yerr=ci_multi, fmt='o', 
             color='#1f77b4', capsize=6, alpha=0.6, elinewidth=1.5)

# Text-only line
plt.plot(training_samples, acc_text, marker='s', linestyle='--', 
         color='#d62728', linewidth=2.5, markersize=8, label='Text-only')
plt.errorbar(training_samples, acc_text, yerr=ci_text, fmt='s', 
             color='#d62728', capsize=6, alpha=0.6, elinewidth=1.5)

# ==========================================
# 4. DATA LABELS & ANNOTATIONS
# ==========================================
for x, y in zip(training_samples, acc_multi):
    plt.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", 
                 xytext=(0, 12), ha='center', color='#1f77b4', fontweight='bold', fontsize=11)

for x, y in zip(training_samples, acc_text):
    plt.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", 
                 xytext=(5, -25), ha='center', color='#d62728', fontweight='bold', fontsize=11)

# ==========================================
# 5. AXIS FORMATTING
# ==========================================
plt.xticks(training_samples, x_labels, fontsize=12)
plt.xlabel('Training Samples', fontsize=13, fontweight='500')
plt.ylabel('Accuracy (%)', fontsize=13, fontweight='500')
plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=False)
plt.ylim(0, 100)
plt.xlim(5000, 55000)

# ==========================================
# 6. SAVE & DISPLAY
# ==========================================
plt.tight_layout()

# Save as PNG (300 DPI for presentations/screens)
plt.savefig('scaling_law_accuracy.png', dpi=300, bbox_inches='tight')

# ✅ Save as PDF (Vector format for thesis/Word insertion)
plt.savefig('scaling_law_accuracy.pdf', bbox_inches='tight')

print("✅ Saved: scaling_law_accuracy.png")
print("✅ Saved: scaling_law_accuracy.pdf")

plt.show()