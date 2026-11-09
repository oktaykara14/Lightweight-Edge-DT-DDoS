import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = os.path.join("figures", "aptos.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)


def configure_plot_style():
    plt.rcParams.update({
        "font.family": "Aptos",
        "font.sans-serif": ["Aptos", "DejaVu Sans"],
        "font.size": 12,
        "savefig.dpi": 600,
        "axes.linewidth": 1.5,
        "figure.facecolor": "#ffffff"
    })


def create_figure_7_performance():
    """Generates Figure 7 comparing detection latency and CPU load across Cloud, Edge, and Proposed DT."""
    print("Generating Figure 7 (Real-Time Efficiency & Computational Overhead Comparison)...")

    labels = ['Cloud Baseline', 'Standard Edge\n(non-optimized)', 'Proposed DT\nFramework']
    latency = [155.0, 58.0, 18.5]
    cpu_load = [10.0, 85.0, 11.6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    colors = ['#b2bec3', '#d63031', '#00b894']

    # Subplot 1: Detection Latency
    bars1 = ax1.bar(labels, latency, color=colors, edgecolor='#2d3436', alpha=0.9, width=0.6, zorder=3)
    ax1.set_ylabel('Detection Latency ($T_{det}$) [ms]', weight='bold', fontsize=14)
    ax1.set_title('Real-Time Efficiency Analysis', weight='black', fontsize=16, pad=20)
    ax1.set_ylim(0, 180)
    ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h} ms",
                 ha='center', va='bottom', weight='black', fontsize=13)

    # Subplot 2: CPU Utilization
    bars2 = ax2.bar(labels, cpu_load, color=colors, edgecolor='#2d3436', alpha=0.9, width=0.6, zorder=3)
    ax2.set_ylabel('CPU Utilization Rate [%]', weight='bold', fontsize=14)
    ax2.set_title('Computational Overhead Comparison', weight='black', fontsize=16, pad=20)
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{h}%",
                 ha='center', va='bottom', weight='black', fontsize=13)

    plt.tight_layout(pad=5)

    save_path = f"{output_dir}/figure7.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Saved figure: {save_path}")


if __name__ == "__main__":
    configure_plot_style()
    create_figure_7_performance()
