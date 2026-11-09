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
        "font.size": 14,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "figure.facecolor": "#ffffff"
    })


def draw_tier_block(ax, x, y, w, h, title, lines, edge_color='#2c3e50'):
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor='#ffffff', edgecolor=edge_color, lw=2, zorder=3))
    header_h = h * 0.22
    ax.add_patch(patches.Rectangle((x, y + h - header_h), w, header_h, facecolor=edge_color, edgecolor=edge_color, zorder=4))
    ax.text(x + w / 2, y + h - header_h / 2, title, ha='center', va='center', weight='bold', color='white', fontsize=21, zorder=5)

    for i, line in enumerate(lines):
        ax.text(x + 0.02, y + h - header_h - 0.05 - (i * 0.06), f"\u2022 {line}", ha='left', va='top', fontsize=19, zorder=5)


def create_figure_4():
    """Generates Figure 4 illustrating the three-tier hierarchical edge-centric DT architecture."""
    fig, ax = plt.subplots(figsize=(24, 18))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(patches.Rectangle((0.01, 0.01), 0.08, 0.98, facecolor='#34495e', edgecolor='none', zorder=1))
    for i, t in enumerate(["TIER 1: PHYSICAL", "TIER 2: EDGE-DT", "TIER 3: SECURITY"]):
        ax.text(0.05, 0.15 + i * 0.35, t, rotation=90, ha='center', va='center', weight='black', color='white', fontsize=26)

    ax.add_patch(patches.Rectangle((0.1, 0.68), 0.88, 0.3, facecolor='#ffffff', edgecolor='#dee2e6', ls='-', zorder=0))
    ax.add_patch(patches.Rectangle((0.1, 0.35), 0.88, 0.3, facecolor='#fdfdfd', edgecolor='#dee2e6', ls='-', zorder=0))
    ax.add_patch(patches.Rectangle((0.1, 0.02), 0.88, 0.3, facecolor='#ffffff', edgecolor='#dee2e6', ls='-', zorder=0))

    bw, bh = 0.40, 0.28

    # Tier 3: Security Orchestration Layer
    draw_tier_block(ax, 0.13, 0.69, bw, bh, "QUANTIZED ML ENGINE",
                    ["Volumetric Anomaly Detection", "8-bit RF Inference Engine", "Tdet < 50ms Detection"], '#1864ab')
    draw_tier_block(ax, 0.56, 0.69, bw, bh, "AUTONOMOUS CONTROLLER",
                    ["Automated Mitigation Trigger", "Alarm-driven State Sync", "Response Policy Dispatch"], '#c92a2a')

    # Tier 2: Edge-Digital Twin Layer
    draw_tier_block(ax, 0.13, 0.36, bw, bh, "LOCALIZED DT INSTANCE",
                    ["Sub-ms State Mirroring", "Event-Driven Sync Feed", "High-Entropy Updates"], '#1864ab')
    draw_tier_block(ax, 0.56, 0.36, bw, bh, "POLICY ENGINE",
                    ["IP Rate Limiting Actions", "Port Blocking Commands", "Local Execution Hub"], '#1864ab')

    # Tier 1: Physical IoT Layer
    draw_tier_block(ax, 0.13, 0.03, bw, bh, "IOT INFRASTRUCTURE",
                    ["Heterogeneous Device Fleet", "MQTT / CoAP Protocols", "Metadata Extraction"], '#495057')
    draw_tier_block(ax, 0.56, 0.03, bw, bh, "ENFORCEMENT LAYER",
                    ["Checked Mitigation Policies", "Firewall Filtering Engine", "Traffic Redirection Hub"], '#c92a2a')

    arrow_props = dict(arrowstyle='-|>', lw=3, color='#34495e', mutation_scale=25)

    ax.annotate('', xy=(0.33, 0.36), xytext=(0.33, 0.31), arrowprops=arrow_props)
    ax.text(0.34, 0.335, "Structural Metadata", ha='left', weight='bold', fontsize=17,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.annotate('', xy=(0.33, 0.69), xytext=(0.33, 0.64), arrowprops=arrow_props)
    ax.text(0.34, 0.665, "Event-Driven Sync", ha='left', weight='bold', fontsize=17,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.annotate('', xy=(0.76, 0.64), xytext=(0.76, 0.69),
                arrowprops=dict(arrowstyle='-|>', lw=3, color='#c92a2a', mutation_scale=25))
    ax.text(0.77, 0.665, "Mitigation Trigger", ha='left', weight='bold', fontsize=17,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.annotate('', xy=(0.76, 0.31), xytext=(0.76, 0.36),
                arrowprops=dict(arrowstyle='-|>', lw=3, color='#c92a2a', ls='--', mutation_scale=25))
    ax.text(0.77, 0.335, "Policy Modification", ha='left', weight='bold', fontsize=17,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    out_path = f"{output_dir}/figure4.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    configure_plot_style()
    create_figure_4()
