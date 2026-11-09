import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import os

font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aptos.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "DejaVu Sans"],
        "font.size": 10,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "figure.facecolor": "#ffffff"
    })


def create_figure_2_evolution():
    """Generates Figure 2 illustrating the paradigm shift toward edge-centric Digital Twins."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    layers = [
        "Quantized, Sub-20ms DDoS Inference (Proactive Defense)",
        "Hierarchical Hybrid Feature Engineering (IG-PCA)",
        "Edge-Based, Asynchronous Digital Twin",
        "Traditional Cybersecurity (Passive Monitoring)"
    ]
    layers.reverse()

    colors = ["#34495e", "#2980b9", "#16a085", "#27ae60"]
    base_x, base_w = 0.2, 0.48
    base_y = 0.18
    h = 0.14
    gap = 0.025

    for i, text in enumerate(layers):
        y = base_y + i * (h + gap)
        ax.add_patch(patches.Rectangle((base_x, y), base_w, h, facecolor=colors[i], edgecolor='none', zorder=2))
        ax.text(base_x + base_w / 2, y + h / 2, text,
                va='center', ha='center', weight='bold', color='white', fontsize=11, zorder=4)

    arrow_x = base_x + base_w + 0.08
    top_y = base_y + 4 * h + 3 * gap
    ax.annotate('', xy=(arrow_x, top_y), xytext=(arrow_x, base_y),
                arrowprops=dict(arrowstyle='-|>', color='#2c3e50', lw=4, mutation_scale=25))

    mid_y = (base_y + top_y) / 2
    text_spacing = 0.05

    ax.text(arrow_x + 0.05, mid_y, "EVOLUTION",
            rotation=270, weight='black', fontsize=16, color='#2c3e50', ha='center', va='center')
    ax.text(arrow_x + 0.05 + text_spacing, mid_y, "Transition to Edge-Optimized Intelligence",
            rotation=270, weight='medium', fontsize=11, color='#7f8c8d', ha='center', va='center')

    out_path = f"{output_dir}/figure2_evolution_final.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    configure_plot_style()
    create_figure_2_evolution()
