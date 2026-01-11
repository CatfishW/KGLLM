import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'medium'
plt.rcParams['font.size'] = 12

# Data points: (Name, calls_avg, latency_avg, hits1_score, color, font_size)
data = [
    # Embedding / GNN - Bottom Left
    ("Embed\nKGQA", -0.7, 0.1, 66.6, "#E5E5E5", 10), 
    ("QA-GNN", -0.7, 1.3, 73.0, "#CCCCCC", 10),
    ("Uni\nKGQA", 0.1, 1.1, 75.1, "#BDBDBD", 10),
    
    # Efficient / Mid-range
    ("G-Retriever", 1.8, 1.3, 82.2, "#FFB366", 11), 
    
    # Heavy Agents - Right Side
    ("ToG", 6.2, 3.2, 76.2, "#FFB3B3", 11),
    ("KG-Agent", 8.2, 4.6, 79.2, "#FF8080", 11),
    ("FiDeLiS", 7.4, 4.0, 84.4, "#D32F2F", 11),
    ("RoG", 4.9, 2.4, 85.7, "#B71C1C", 11),
    
    # Ours
    ("APR\n(Ours)", 0.8, 0.5, 85.9, "#66FF66", 13),
]

names = [d[0] for d in data]
x = [d[1] for d in data]
y = [d[2] for d in data]
z = [d[3] for d in data]
colors = [d[4] for d in data]
fs = [d[5] for d in data]

fig, ax = plt.subplots(figsize=(9, 5.5))

# Scaling function for bubble sizes
def get_size(score):
    base = score - 45
    if base < 0: base = 0
    return (base ** 2.2) * 1.6

sizes = [get_size(s) for s in z]

# Plot Logic
for i in range(len(data)):
    text_color = 'black'
    if colors[i] in ["#D32F2F", "#B71C1C"]:
        text_color = 'white'
        
    zo = 3
    edge_width = 1.2
    alpha = 0.9
    
    if "APR" in names[i]:
        zo = 10
        edge_width = 2.5
        
    ax.scatter(x[i], y[i], s=sizes[i], c=colors[i], alpha=alpha, edgecolors='black', linewidth=edge_width, zorder=zo)
    
    label_text = names[i]
    if z[i] > 80 or "APR" in names[i]:
        label_text += f"\n{z[i]}%"
        
    weight = 'bold' if ("APR" in names[i] or z[i] > 84) else 'semibold'
    
    ax.text(x[i], y[i], label_text, 
            ha='center', va='center', 
            fontsize=fs[i], fontweight=weight, color=text_color, zorder=zo+1)

# Regions
ax.axvspan(2.8, 9.8, color='#FFF8F8', alpha=0.6, zorder=0)
ax.text(6.3, 5.5, "Agentic Paradigm\n(High Latency & Cost)", ha='center', va='center', color='#B71C1C', fontweight='bold', fontsize=14)

ax.axvspan(-1.3, 2.3, color='#F5FFF5', alpha=0.6, zorder=0)
ax.text(0.5, 5.5, "Efficient Paradigm\n(Low Latency)", ha='center', va='center', color='#2E7D32', fontweight='bold', fontsize=14)

# Axis labels
ax.set_xlabel('Avg. Number of LLM Calls', fontsize=15, fontweight='bold')
ax.set_ylabel('Inference Latency (Seconds)', fontsize=15, fontweight='bold')

ax.grid(True, linestyle=':', alpha=0.5)
ax.set_ylim(-0.7, 6.2)
ax.set_xlim(-1.3, 9.5)

ax.tick_params(axis='both', which='major', labelsize=13)

# --------------------------
# MANUAL LEGEND (Custom drawn)
# --------------------------
# Refined coordinates for better bottom-right positioning
leg_x_start = 5.8
leg_y_start = -0.55
leg_width = 3.5
leg_height = 2.0

rect = patches.FancyBboxPatch((leg_x_start, leg_y_start), leg_width, leg_height, boxstyle="round,pad=0.08", 
                             linewidth=1.0, edgecolor='#999999', facecolor='white', alpha=0.98, zorder=15)
ax.add_patch(rect)

# Legend Title
title_y = leg_y_start + leg_height - 0.32
ax.text(leg_x_start + leg_width/2, title_y, "Accuracy Scale", ha='center', va='center', fontweight='bold', fontsize=12, zorder=20)

# Legend Bubbles - use smaller scaled sizes to avoid overlap
leg_scores = [65, 75, 85]
leg_x_positions = [leg_x_start + 0.6, leg_x_start + 1.7, leg_x_start + 2.9]
leg_y_bubbles = leg_y_start + 0.8

# Smaller legend bubble sizes (scale down for legend display)
def get_legend_size(score):
    base = score - 45
    if base < 0: base = 0
    return (base ** 2.2) * 0.8  # Half the size of main bubbles

for score, lx in zip(leg_scores, leg_x_positions):
    sz = get_legend_size(score)
    ax.scatter(lx, leg_y_bubbles, s=sz, c='#F5F5F5', alpha=0.95, edgecolors='#555555', linewidth=0.8, zorder=20)
    ax.text(lx, leg_y_bubbles, f"{score}%", ha='center', va='center', fontsize=9, fontweight='semibold', zorder=21)

plt.tight_layout()
plt.savefig('figure1_tradeoff.png', dpi=300)
print("Figure 1 generated successfully as figure1_tradeoff.png")