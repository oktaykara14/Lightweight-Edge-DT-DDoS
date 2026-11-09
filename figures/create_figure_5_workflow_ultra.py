import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import os

font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aptos.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "DejaVu Sans"],
        "font.size": 14,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "figure.facecolor": "#ffffff"
    })


def draw_stage_box(ax, x, y, w, h, title, text=None, color='#2c3e50', alpha=0.1):
    ax.add_patch(patches.Rectangle((x + 0.005, y - 0.005), w, h, color='gray', alpha=0.05, zorder=1))
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                  facecolor=color, alpha=alpha, edgecolor=color, lw=1.5, zorder=2)
    ax.add_patch(rect)
    header_h = 0.045
    ax.add_patch(patches.FancyBboxPatch((x, y + h - header_h), w, header_h, boxstyle="round,pad=0.01",
                                        facecolor=color, alpha=0.8, edgecolor=color, zorder=3))
    ax.text(x + w / 2, y + h - header_h / 2, title, ha='center', va='center', weight='bold', color='white', fontsize=13, zorder=4)
    if text:
        ax.text(x + w / 2, y + (h - header_h) / 2, text, ha='center', va='center', fontsize=12, linespacing=1.4, zorder=4)


def create_figure_5_workflow():
    """Generates Figure 5 depicting the dual-tier port-free hybrid IG-PCA feature pipeline."""
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_stage_box(ax, 0.02, 0.4, 0.16, 0.22, "RAW INPUT", "Network Traffic\n(78 Features)\nPCAP Stream", color='#7f8c8d')

    arrow_props = dict(arrowstyle='->', lw=2.5, color='#34495e', connectionstyle="arc3,rad=-0.1")
    ax.annotate('', xy=(0.23, 0.65), xytext=(0.18, 0.52), arrowprops=arrow_props)
    ax.annotate('', xy=(0.23, 0.35), xytext=(0.18, 0.48),
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495e', connectionstyle="arc3,rad=0.1"))

    draw_stage_box(ax, 0.23, 0.55, 0.35, 0.32, "STAGE 1: IG FILTERING",
                   "Selection of 8 Flow/Packet Headers:\n• Protocol, Flow Duration, Inbound\n• Tot. Fwd/Bwd Packets\n• Flow Packets/s & Flow Bytes/s\n• Average Packet Size",
                   color='#27ae60')

    draw_stage_box(ax, 0.23, 0.13, 0.35, 0.32, "STAGE 2: PCA ENGINE",
                   "Processing 70 Residual Features:\n• Covariance Matrix Comp.\n• Eigenvector Projection\n• 7 Principal Components\n(98.1% Variance Retained)",
                   color='#2980b9')

    ax.annotate('', xy=(0.63, 0.5), xytext=(0.58, 0.7), arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495e'))
    ax.annotate('', xy=(0.63, 0.5), xytext=(0.58, 0.3), arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495e'))

    draw_stage_box(ax, 0.63, 0.4, 0.16, 0.22, "HYBRID VECTOR", "15 Components:\n(8 Primary +\n7 Secondary)", color='#f39c12')

    ax.annotate('', xy=(0.84, 0.51), xytext=(0.79, 0.51), arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495e'))

    draw_stage_box(ax, 0.84, 0.4, 0.14, 0.22, "FINAL ENGINE", "Min-Max Norm [0, 1]\n\nQuantized RF\nClassifier (8-bit)", color='#c0392b')

    ax.text(0.5, 0.05, "Dimensionality Reduction: 80.7% (78 \u2192 15) | Feature-Space & Traversal Cost Savings: ~82% | Thermal Throttling Prevention",
            ha='center', weight='bold', color='#2c3e50', fontsize=13)

    out_path = f"{output_dir}/figure5.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    configure_plot_style()
    create_figure_5_workflow()
