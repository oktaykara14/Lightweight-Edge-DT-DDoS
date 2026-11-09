import os
import numpy as np
import matplotlib.pyplot as plt

output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

def create_figure_3_radar():
    """Generates Figure 3 comparing the proposed DT framework against a traditional cloud IDS across security standards."""
    labels = [
        'IoT Security (NIST 800-82)',
        'IT Data Privacy (GDPR)',
        'OT Operational Resilience',
        'Industrial Compliance (ISO 27001)',
        'Network Availability (CIA)'
    ]

    proposed_values = [97.0, 92.0, 88.0, 95.0, 98.0]
    traditional_values = [62.0, 55.0, 48.0, 65.0, 72.0]

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    proposed_values += proposed_values[:1]
    traditional_values += traditional_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.fill(angles, proposed_values, color='teal', alpha=0.3, label='Proposed DT-DDoS Framework')
    ax.plot(angles, proposed_values, color='teal', linewidth=3, marker='o')

    ax.fill(angles, traditional_values, color='gray', alpha=0.2, label='Traditional IoT System')
    ax.plot(angles, traditional_values, color='gray', linewidth=2, linestyle='--', marker='s')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold', wrap=True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.subplots_adjust(top=0.85, bottom=0.15)

    out_path = f"{output_dir}/figure3.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    create_figure_3_radar()
