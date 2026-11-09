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
        "font.size": 13,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "figure.facecolor": "#ffffff"
    })


def create_figure_8_swot():
    """Generates Figure 8 illustrating the strategic SWOT analysis of the proposed DT-DDoS framework."""
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    quadrants = [
        {
            "pos": (0.05, 0.52), "color": "#27ae60", "title": "STRENGTHS",
            "items": [
                "ARM-based Gateway Optimization",
                "Hybrid IG-PCA Dimensionality Reduction",
                "8-bit Quantized Random Forest",
                "Low Streaming Footprint (206.8 MB RAM)",
                "Prevention of Thermal Throttling"
            ]
        },
        {
            "pos": (0.52, 0.52), "color": "#f39c12", "title": "WEAKNESSES",
            "items": [
                "Dataset-Specific Bias (CICDDoS2019)",
                "Theoretical Risk of Overfitting",
                "Abstraction of Raw Packet Attributes",
                "Payload-based Identification Challenges",
                "Need for Multi-domain Validation"
            ]
        },
        {
            "pos": (0.05, 0.05), "color": "#2980b9", "title": "OPPORTUNITIES",
            "items": [
                "6G-Enabled Industrial IoT Ecosystems",
                "Real-Time Edge-to-Mirror Decision Sync",
                "Distributed Blockchain for Trust Mgmt",
                "Scalable 6G Edge of Things Architecture",
                "Autonomous Threat Response Integration"
            ]
        },
        {
            "pos": (0.52, 0.05), "color": "#c0392b", "title": "THREATS",
            "items": [
                "Adversarial Machine Learning (AML)",
                "Evasion Attacks via Decision Perturbation",
                "Cross-Layer Information Leakage",
                "False Data Injection in Control Loops",
                "Evolving Volumetric Attack Patterns"
            ]
        }
    ]

    for q in quadrants:
        x, y = q["pos"]
        ax.add_patch(patches.Rectangle((x + 0.005, y - 0.005), 0.43, 0.43, color='gray', alpha=0.1, zorder=1))
        rect = patches.FancyBboxPatch((x, y), 0.43, 0.43, boxstyle="round,pad=0.01",
                                      facecolor=q["color"], alpha=0.1, edgecolor=q["color"], lw=2, zorder=2)
        ax.add_patch(rect)

        header = patches.FancyBboxPatch((x, y + 0.38), 0.43, 0.05, boxstyle="round,pad=0.01",
                                        facecolor=q["color"], alpha=0.9, edgecolor=q["color"], zorder=3)
        ax.add_patch(header)
        ax.text(x + 0.215, y + 0.405, q["title"], ha='center', va='center', weight='bold', color='white', fontsize=15, zorder=4)

        for i, item in enumerate(q["items"]):
            ax.text(x + 0.03, y + 0.34 - (i * 0.065), f"• {item}", ha='left', va='center', fontsize=13, zorder=4)

    circle = plt.Circle((0.5, 0.5), 0.07, color='white', ec='#2c3e50', lw=2, zorder=5)
    ax.add_artist(circle)
    ax.text(0.5, 0.5, "SWOT", ha='center', va='center', weight='bold', fontsize=17, color='#2c3e50', zorder=6)

    out_path = f"{output_dir}/figure8_swot.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    configure_plot_style()
    create_figure_8_swot()
