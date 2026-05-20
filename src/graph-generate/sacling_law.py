import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. EXPERIMENTAL DATA (From your results)
# ==========================================
N_train = np.array([10000, 30000, 50000])
loss_text = np.array([2.684183, 1.053350, 0.792949])
loss_multi = np.array([1.016064, 0.459560, 0.417141])
acc_text = np.array([23.04, 72.45, 77.94])
acc_multi = np.array([77.52, 87.84, 88.78])

# ==========================================
# 2. SCALING LAW FUNCTIONS & FITTING
# ==========================================
def loss_scaling(N, A, alpha, B):
    return A * (N ** -alpha) + B

def acc_scaling(N, C, D, beta):
    return C - D * (N ** -beta)

# Fit Loss Scaling
p0_loss_t = [1e7, 2.0, 0.7]
p0_loss_m = [1e9, 2.0, 0.3]
popt_loss_t, _ = curve_fit(loss_scaling, N_train, loss_text, p0=p0_loss_t, maxfev=5000, bounds=([0, 0, 0], [np.inf, 5, 1.0]))
popt_loss_m, _ = curve_fit(loss_scaling, N_train, loss_multi, p0=p0_loss_m, maxfev=5000, bounds=([0, 0, 0], [np.inf, 5, 1.0]))
A_t, a_t, B_t = popt_loss_t
A_m, a_m, B_m = popt_loss_m

# Fit Accuracy Scaling
p0_acc_t = [80.0, 1e5, 1.5]
p0_acc_m = [92.0, 5e4, 1.0]
popt_acc_t, _ = curve_fit(acc_scaling, N_train, acc_text, p0=p0_acc_t, maxfev=5000, bounds=([0, 0, 0], [100, np.inf, 5]))
popt_acc_m, _ = curve_fit(acc_scaling, N_train, acc_multi, p0=p0_acc_m, maxfev=5000, bounds=([0, 0, 0], [100, np.inf, 5]))
C_t, D_t, b_t = popt_acc_t
C_m, D_m, b_m = popt_acc_m

# Generate smooth curves for plotting
N_smooth = np.logspace(3.8, 5.5, 200)
loss_t_fit = loss_scaling(N_smooth, *popt_loss_t)
loss_m_fit = loss_scaling(N_smooth, *popt_loss_m)
acc_t_fit = acc_scaling(N_smooth, *popt_acc_t)
acc_m_fit = acc_scaling(N_smooth, *popt_acc_m)

# Compute residuals
loss_t_pred = loss_scaling(N_train, *popt_loss_t)
loss_m_pred = loss_scaling(N_train, *popt_loss_m)
res_t = loss_text - loss_t_pred
res_m = loss_multi - loss_m_pred

# Compute data efficiency (samples needed for target accuracy)
targets = np.array([75.0, 80.0, 85.0])
def inv_acc_scaling(acc, C, D, beta):
    return ((C - acc) / D) ** (-1.0 / beta)

N_t_targets = inv_acc_scaling(targets, C_t, D_t, b_t)
N_m_targets = inv_acc_scaling(targets, C_m, D_m, b_m)

# ==========================================
# 3. ACADEMIC PLOTTING CONFIGURATION
# ==========================================
plt.rcParams.update({
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': '600',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.color': '#EAECEE',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Scaling Law Analysis: Multimodal vs. Text-Only Document Classification", 
             fontsize=13, fontweight='bold', y=0.995)

colors = {'text': '#C0392B', 'multi': '#1B4F72'}

# --------------------------------------------------
# PANEL (a): Loss Scaling (Log-Log)
# --------------------------------------------------
ax = axes[0, 0]
ax.scatter(N_train, loss_text, c=colors['text'], s=65, marker='s', label='Text-only (data)', zorder=5, edgecolor='k')
ax.scatter(N_train, loss_multi, c=colors['multi'], s=65, marker='o', label='Multimodal (data)', zorder=5, edgecolor='k')
ax.plot(N_smooth, loss_t_fit, color=colors['text'], linestyle='--', linewidth=1.5, label='Text-only fit')
ax.plot(N_smooth, loss_m_fit, color=colors['multi'], linestyle='-', linewidth=1.5, label='Multimodal fit')
ax.axhline(y=B_t, color=colors['text'], linestyle=':', alpha=0.7, label=f'Text asymptote (B={B_t:.3f})')
ax.axhline(y=B_m, color=colors['multi'], linestyle=':', alpha=0.7, label=f'Multi asymptote (B={B_m:.3f})')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Training Samples (N)')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('(a) Loss Scaling: L(N) = A·N^(-α) + B')
ax.legend(loc='upper right', frameon=True, edgecolor='gray')

# --------------------------------------------------
# PANEL (b): Accuracy Scaling (Semi-Log)
# --------------------------------------------------
ax = axes[0, 1]
ax.scatter(N_train, acc_text, c=colors['text'], s=65, marker='s', label='Text-only', zorder=5, edgecolor='k')
ax.scatter(N_train, acc_multi, c=colors['multi'], s=65, marker='o', label='Multimodal', zorder=5, edgecolor='k')
ax.plot(N_smooth, acc_t_fit, color=colors['text'], linestyle='--', linewidth=1.5)
ax.plot(N_smooth, acc_m_fit, color=colors['multi'], linestyle='-', linewidth=1.5)
ax.axhline(y=C_t, color=colors['text'], linestyle=':', alpha=0.6, label=f'Text max (C={C_t:.1f}%)')
ax.axhline(y=C_m, color=colors['multi'], linestyle=':', alpha=0.6, label=f'Multi max (C={C_m:.1f}%)')
ax.set_xscale('log')
ax.set_ylim(0, 100)
ax.set_xlabel('Training Samples (N)')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('(b) Accuracy Scaling: ACC(N) = C - D·N^(-β)')
ax.legend(loc='lower right', frameon=True, edgecolor='gray')

# --------------------------------------------------
# PANEL (c): Residual Analysis
# --------------------------------------------------
ax = axes[1, 0]
ax.scatter(loss_t_pred, res_t, c=colors['text'], s=65, label='Text-only', edgecolor='k', zorder=5)
ax.scatter(loss_m_pred, res_m, c=colors['multi'], s=65, label='Multimodal', edgecolor='k', zorder=5)
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('Predicted Loss')
ax.set_ylabel('Residual (Observed - Predicted)')
ax.set_title('(c) Residual Analysis')
ax.legend(loc='upper right', frameon=True, edgecolor='gray')

# --------------------------------------------------
# PANEL (d): Data Efficiency (Horizontal Bar Chart)
# --------------------------------------------------
ax = axes[1, 1]
y_pos = np.arange(len(targets))
bar_height = 0.35

# Plot multimodal (shorter bars)
ax.barh(y_pos + bar_height/2, N_m_targets, height=bar_height, color=colors['multi'], alpha=0.85, 
        label='Multimodal', edgecolor='black', linewidth=1.2, zorder=3)
# Plot text-only (longer bars) behind
ax.barh(y_pos - bar_height/2, N_t_targets, height=bar_height, color=colors['text'], alpha=0.75, 
        label='Text-only', edgecolor='black', linewidth=1.2, zorder=2)

ax.set_yticks(y_pos)
ax.set_yticklabels([f'{t}%' for t in targets])
ax.set_xlabel('Training Samples Needed')
ax.set_ylabel('Target Accuracy')
ax.set_xscale('log')
ax.set_title('(d) Data Efficiency: Samples to Reach Target')
ax.legend(loc='lower right', frameon=True, edgecolor='gray')
ax.invert_yaxis()  # 85% at top, matching reference

# --------------------------------------------------
# FINAL LAYOUT & EXPORT
# --------------------------------------------------
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save outputs
for ext in ['.png', '.pdf']:
    plt.savefig(f'scaling_law_comparison{ext}', dpi=300 if ext=='.png' else None, bbox_inches='tight')
plt.close()

print("✅ Scaling law comparison figure saved as: scaling_law_comparison.png & .pdf")

# Print fitted parameters for your report
print("\n" + "="*60)
print("FITTED SCALING PARAMETERS (For Section 4.1.3)")
print("="*60)
print(f"Text-only Loss:    L(N) = {A_t:.2e}·N^(-{a_t:.3f}) + {B_t:.3f}")
print(f"Multimodal Loss:   L(N) = {A_m:.2e}·N^(-{a_m:.3f}) + {B_m:.3f}")
print(f"Text-only ACC:     ACC(N) = {C_t:.2f} - {D_t:.2e}·N^(-{b_t:.3f})")
print(f"Multimodal ACC:    ACC(N) = {C_m:.2f} - {D_m:.2e}·N^(-{b_m:.3f})")
print("="*60)