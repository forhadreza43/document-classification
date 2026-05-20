import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# ACADEMIC COLOR PALETTE
# ==========================================
PALETTE = {
    'primary': '#1B4F72',      # Deep Navy (n₀₁)
    'secondary': '#C0392B',    # Refined Crimson (n₁₀)
    'accent': "#3596C0",       # Professional Blue (annotations)
    'pie_1': "#00D6D6",        # Cyan
    'pie_2': "#C5C209",        # Gold-Yellow
    'pie_3': '#D4AC0D',        # Academic Gold
    'text': "#4C6D8F",         # Dark Slate
    'grid': '#EAECEE',         # Soft Grid
    'bg': '#FFFFFF'
}

# ==========================================
# YOUR ACTUAL RESULTS
# ==========================================
n01 = 4966
n10 = 53
net_gain = n01 - n10

# PLACEHOLDER: Update with your actual breakdown
breakdown_labels = ['Layout-Dependent Docs', 'Visual/Structural Cues', 'Purely Textual Docs']
breakdown_values = [3200, 1100, 666]
pie_colors = [PALETTE['pie_1'], PALETTE['pie_2'], PALETTE['pie_3']]

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titleweight': '600',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
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
    'axes.edgecolor': '#D5D8DC'
})

# ==========================================
# FIGURE SETUP
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), 
                               gridspec_kw={'width_ratios': [1.3, 1]})

# --------------------------------------------------
# LEFT PANEL: Discordant Pairs Bar Chart
# --------------------------------------------------
categories = ['Multimodal Corrects\nText Errors (n_01)', 
              'Text Corrects\nMultimodal Errors (n_10)']
values = [n01, n10]
bar_colors = [PALETTE['primary'], PALETTE['secondary']]

bars = ax1.bar(categories, values, color=bar_colors, edgecolor='white', 
               linewidth=2, width=0.42, zorder=3)

ax1.set_ylim(0, n01 * 1.15)
ax1.set_ylabel('Number of Documents', color=PALETTE['text'], fontweight='500', labelpad=10)
ax1.tick_params(axis='x', rotation=0, labelsize=11)
ax1.yaxis.grid(True)
ax1.set_axisbelow(True)

# Value annotations
for bar, val in zip(bars, values):
    ax1.annotate(f'{val:,}',
                 xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                 xytext=(0, 8), textcoords='offset points',
                 ha='center', va='bottom',
                 fontsize=16, fontweight='bold', color=PALETTE['text'])

# Net gain connector
arrow_y = n01 * 0.48
ax1.annotate('', xy=(0, arrow_y), xytext=(1, arrow_y),
             arrowprops=dict(arrowstyle='<->', color=PALETTE['accent'], lw=2.5, shrinkA=8, shrinkB=8))
ax1.text(0.5, arrow_y + n01*0.03, f'Net Gain: +{net_gain:,} Documents', 
         ha='center', va='bottom', fontsize=16, fontweight='600', color=PALETTE['accent'],
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F4F6F7', edgecolor=PALETTE['accent'], alpha=0.95, lw=1.2))

# --------------------------------------------------
# RIGHT PANEL: Category Breakdown (Donut Chart) - FIXED
# --------------------------------------------------
def format_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return f'{pct:.0f}%\n({val:,})'
    return my_autopct

# KEY FIXES for pie chart text overflow:
# 1. Reduce pctdistance (brings % text closer to center)
# 2. Reduce labeldistance (brings category labels closer)
# 3. Increase donut width (more ring space for text)
# 4. Use smaller fontsize for autopct vs labels
# 5. Add bbox background for better readability

wedges, texts, autotexts = ax2.pie(
    breakdown_values,
    labels=breakdown_labels,
    autopct=format_autopct(breakdown_values),
    colors=pie_colors,
    startangle=95,
    radius=1.0,                          # ← Control overall pie size
    wedgeprops=dict(width=0.65, edgecolor='white', linewidth=2.5),  # ← Wider ring = more text space
    textprops=dict(fontsize=16, weight='500', color=PALETTE['text']),  # ← Smaller label font
    pctdistance=0.75,                    # ← Closer to center (was 0.82)
    labeldistance=1.08,                  # ← Closer to pie edge (was 1.15)
    rotatelabels=False
)

# Style percentage text (autopct) - keep smaller to fit in slices
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)            # ← Smaller than labels to fit in slices
    autotext.set_color('white')          # ← White text for contrast on colored slices

# Style category labels (outside the pie)
for text in texts:
    text.set_fontsize(16)                # ← Keep labels readable but not oversized
    text.set_fontweight('500')

# Add center white circle for donut effect (adjusted for wider ring)
centre_circle = plt.Circle((0, 0), 0.35, fc=PALETTE['bg'], linewidth=2, edgecolor='#E5E8E8')
ax2.add_patch(centre_circle)

ax2.set_title('Category Breakdown of\nMultimodal Improvements', 
              fontsize=16, fontweight='600', color=PALETTE['text'], pad=15)

# --------------------------------------------------
# FINAL LAYOUT & EXPORT
# --------------------------------------------------
plt.tight_layout(rect=[0, 0.03, 1, 0.92])

output_dir = "figures_disagreement"
os.makedirs(output_dir, exist_ok=True)

for ext in ['.png', '.pdf']:
    save_path = os.path.join(output_dir, f'disagreement_professional_50k{ext}')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=PALETTE['bg'])
plt.close()

print(f"✅ Professional chart saved to '{output_dir}/'")