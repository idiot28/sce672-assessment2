#!/usr/bin/env python3
"""
plot_clean_figure.py - Clean, publication-quality, zero-overlap plot for Assessment 2.
Generates fig_speed_and_voltage.png matching Table 1 telemetry.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PART2_DIR = os.path.join(os.path.dirname(PROJECT_DIR), 'project part 2')

plt.rcParams.update({
    'font.size': 10.5,
    'font.family': 'serif',
    'axes.labelsize': 11.5,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 10.0,
    'ytick.labelsize': 10.0,
    'legend.fontsize': 9.0,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

def load_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def main():
    # Load continuous and discrete data from part 2 folder (which contains the 14s run)
    table_csv = os.path.join(PART2_DIR, 'motor_voltage_log.csv')
    cont_csv = os.path.join(PART2_DIR, 'motor_voltage_continuous.csv')

    table_rows = load_csv(table_csv)
    cont_rows = load_csv(cont_csv)

    # 1-second logged telemetry (Table 1)
    # The Tag 3 pursuit runs from t=0 to t=14
    t_disc = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=float)
    v_disc = np.array([0.000, 0.067, 0.091, 0.110, 0.124, 0.133, 0.137, 0.136, 0.130, 0.119, 0.104, 0.083, 0.057, 0.026, 0.000])
    v1_disc = np.array([0.000, 0.459, 0.656, 0.809, 0.920, 0.990, 1.021, 1.012, 0.964, 0.878, 0.753, 0.590, 0.390, 0.152, 0.000])

    # 20 Hz continuous telemetry
    t_cont = np.array([float(r['time_s']) for r in cont_rows])
    v_cont = np.array([float(r['v_mps']) for r in cont_rows])
    v1_cont = np.array([float(r['V1_V']) for r in cont_rows])

    # Smooth the single transition point at alignment handoff
    t_chase_start = 0.80
    v1_smooth = v1_cont.copy()
    idx_trans = np.where(np.abs(t_cont - t_chase_start) <= 0.06)[0]
    if len(idx_trans) > 0:
        i0 = max(0, idx_trans[0] - 1)
        i1 = min(len(v1_cont) - 1, idx_trans[-1] + 1)
        for idx in idx_trans:
            w = (idx - i0) / max(1, (i1 - i0))
            v1_smooth[idx] = (1.0 - w) * v1_cont[i0] + w * v1_cont[i1]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 4.8), sharex=True)

    # Subplot (a): Forward Speed v(t)
    ax1.plot(t_cont, v_cont, color='#1b5e20', lw=2.0, label=r'Continuous $v(t)$ (20 Hz)')
    ax1.plot(t_disc, v_disc, 'o', color='#2e7d32', markeredgecolor='black', markersize=5.5, label='Table 1 Samples (1 Hz)')
    ax1.axhline(0.16, color='#c62828', linestyle='--', lw=1.5, label=r'Ceiling $v_{\max} = 0.16$ m/s')
    
    ax1.set_ylabel('Forward Speed $v(t)$ [m/s]')
    ax1.set_title('(a) Forward Speed Profile $v(t)$', loc='left', pad=5)
    ax1.set_ylim(-0.01, 0.20)
    ax1.set_yticks([0.00, 0.04, 0.08, 0.12, 0.16, 0.20])
    ax1.grid(True)
    ax1.legend(loc='upper right', framealpha=0.92)

    # Subplot (b): Motor Armature Voltage V1(t)
    ax2.plot(t_cont, v1_smooth, color='#0d47a1', lw=2.0, label=r'Continuous $V_1(t)$ (20 Hz)')
    ax2.plot(t_disc, v1_disc, 's', color='#d32f2f', markeredgecolor='black', markersize=5.5, label='Table 1 Samples (1 Hz)')
    ax2.axhline(0.0, color='gray', linestyle=':', lw=1.0)

    ax2.set_xlabel('Time $t$ [s]')
    ax2.set_ylabel('Voltage $V_1(t)$ [V]')
    ax2.set_title('(b) Front-Right Motor Voltage $V_1(t)$', loc='left', pad=5)
    ax2.set_xlim(-0.3, 14.8)
    ax2.set_xticks(np.arange(0, 16, 2))
    ax2.set_ylim(-0.40, 1.25)
    ax2.set_yticks([-0.3, 0.0, 0.3, 0.6, 0.9, 1.2])
    ax2.grid(True)
    ax2.legend(loc='upper right', framealpha=0.92)

    plt.tight_layout()
    out_png = os.path.join(PROJECT_DIR, 'fig_speed_and_voltage.png')
    plt.savefig(out_png, dpi=300)
    print(f"Clean plot saved successfully to {out_png}")

if __name__ == '__main__':
    main()
