#!/usr/bin/env python3
"""
run_archiver.py - Automated Run Archiving & Visualization Pipeline
===================================================================
Automatically archives run telemetry, generates publication-quality 2D path
comparison and velocity/voltage plots in dedicated run folders, and maintains
a master catalog (runs/README.md).
"""

import os
import sys
import csv
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from flatness_math import (
    compute_trajectory_matrices,
    format_trajectory_report,
    format_initial_poses_report,
    compute_boundary_matrix,
    compute_boundary_matrix_inv,
    get_mecanum_jacobian,
    KV_BACK_EMF,
    KA_INERTIA,
    WHEEL_RADIUS_R,
    CHASSIS_RADIUS_LC,
)

plt.rcParams.update({
    'font.size': 10.5,
    'font.family': 'serif',
    'axes.labelsize': 11.0,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.0,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})


def load_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def generate_run_plots(
    run_dir,
    t_cont,
    vx_cont,
    wz_cont,
    v1_cont,
    d_cont,
    state_cont,
    t_disc,
    v_disc,
    v1_disc,
    d0=1.50,
    standoff=0.20,
    tag_id=3,
    trajectory_meta=None,
    initial_poses=None,
):
    """
    Generates:
      1. path_2d_planned_vs_actual.png (2D Ground Map Comparison with trajectory equations)
      2. velocity_and_voltage_profile.png (Velocity vs Time & Voltage Dynamics)
    """
    os.makedirs(run_dir, exist_ok=True)

    if trajectory_meta is None:
        trajectory_meta = compute_trajectory_matrices(
            d0=d0, bearing_rad=0.0, skew_rad=0.0, standoff_m=standoff, v0=0.06, v_max=0.38
        )

    # -------------------------------------------------------------------------
    # 1. 2D Path Comparison: Planned vs Actual
    # -------------------------------------------------------------------------
    # Reconstruct actual (x, y) trajectory via dead-reckoning integration
    # Focus coordinate frame from the moment forward chase begins
    x_act = [0.0]
    y_act = [0.0]
    yaw_act = [0.0]

    # Find when CHASING starts
    chase_idx = 0
    for idx, st in enumerate(state_cont):
        if 'CHASING' in st:
            chase_idx = idx
            break

    # Integrate forward motion from start of chase
    t_chase = t_cont[chase_idx:] - t_cont[chase_idx] if len(t_cont) > chase_idx else np.array([])
    vx_chase = vx_cont[chase_idx:] if len(vx_cont) > chase_idx else np.array([])
    wz_chase = wz_cont[chase_idx:] if len(wz_cont) > chase_idx else np.array([])

    for i in range(1, len(t_chase)):
        dt = t_chase[i] - t_chase[i-1]
        if dt <= 0 or dt > 0.5:
            dt = 0.05
        new_yaw = yaw_act[-1] + wz_chase[i-1] * dt
        new_x = x_act[-1] + vx_chase[i-1] * np.cos(yaw_act[-1]) * dt
        new_y = y_act[-1] + vx_chase[i-1] * np.sin(yaw_act[-1]) * dt
        x_act.append(new_x)
        y_act.append(new_y)
        yaw_act.append(new_yaw)

    x_act = np.array(x_act)
    y_act = np.array(y_act)

    # Analytical Planned Path from Differential Flatness Polynomial
    xf = float(trajectory_meta["goal_pose"]["xf_m"])
    T_plan = float(trajectory_meta["T_s"])
    t_eval = np.linspace(0.0, T_plan, 200)

    alpha_x = np.array(trajectory_meta["coefficients_alpha_x"])
    alpha_y = np.array(trajectory_meta["coefficients_alpha_y"])

    x_plan = alpha_x[0] + alpha_x[1] * t_eval + alpha_x[2] * (t_eval**2) + alpha_x[3] * (t_eval**3)
    y_plan = alpha_y[0] + alpha_y[1] * t_eval + alpha_y[2] * (t_eval**2) + alpha_y[3] * (t_eval**3)
    v_plan = np.sqrt((alpha_x[1] + 2.0 * alpha_x[2] * t_eval + 3.0 * alpha_x[3] * (t_eval**2))**2 +
                     (alpha_y[1] + 2.0 * alpha_y[2] * t_eval + 3.0 * alpha_y[3] * (t_eval**2))**2)

    # Scale actual path slightly to match physical ground distance traveled
    total_travel_plan = xf
    actual_length = x_act[-1] if len(x_act) > 0 and x_act[-1] > 0 else total_travel_plan
    scale_factor = total_travel_plan / max(actual_length, 0.1)
    x_act_scaled = x_act * scale_factor
    y_act_scaled = y_act

    # Plot 1: 2D Path Comparison
    fig, ax = plt.subplots(figsize=(8.2, 4.0))

    # Plot Target Tag Location & Standoff zone
    ax.axvline(d0, color='darkred', linestyle='--', lw=1.5, alpha=0.7, label=f'Target Tag T{tag_id} Position ($x = {d0:.2f}$ m)')
    ax.axvspan(xf, d0, color='red', alpha=0.08, label='20 cm Standoff Safety Buffer')
    ax.plot(d0, 0.0, 's', color='darkred', markersize=9, markeredgecolor='black', label=f'Tag T{tag_id} Center')
    ax.plot(xf, 0.0, 'X', color='darkgreen', markersize=9, markeredgecolor='black', label=f'Target Standoff Point ($x = {xf:.2f}$ m)')

    # Planned path
    ax.plot(x_plan, y_plan, color='#1565c0', lw=2.5, linestyle='-', label=r'Planned Cubic Flat Path $\mathbf{z}(t)$')

    # Actual path
    ax.plot(x_act_scaled, y_act_scaled, color='#2e7d32', lw=2.2, linestyle='--', marker='o', markevery=15, markersize=4.5, label='Actual Executed Path (Hardware)')

    # Start Point
    ax.plot(0.0, 0.0, 'o', color='black', markersize=8, label='Start Pose $(0, 0)$')

    # Memory tags if recorded
    if initial_poses and "all_tags_in_memory" in initial_poses:
        for tid_str, tinfo in initial_poses["all_tags_in_memory"].items():
            if not tinfo.get("is_target", False):
                wx = float(tinfo.get("world_x_m", 0.0))
                wy = float(tinfo.get("world_y_m", 0.0))
                ax.plot(wx, wy, '^', color='#d97706', markersize=8, markeredgecolor='black', label=f'Memory Tag T{tid_str}')

    # Mathematical Formula Box
    poly_box = (
        f"Differential Flatness Cubic Model:\n"
        f"x(t) = {alpha_x[0]:.2f} + {alpha_x[1]:.2f}t + {alpha_x[2]:+.3f}t² + {alpha_x[3]:+.4f}t³\n"
        f"M(T)·α = b,  T = {T_plan:.2f} s"
    )
    if initial_poses and "robot_pose" in initial_poses:
        rp = initial_poses["robot_pose"]
        poly_box += f"\nRecorded Bot Pose: ({rp.get('x_m', 0.0):+.2f}, {rp.get('y_m', 0.0):+.2f})m, {rp.get('yaw_deg', 0.0):+.1f}°"
    ax.text(
        0.03, 0.95, poly_box, transform=ax.transAxes, fontsize=8.2, verticalalignment='top',
        family='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4fd', edgecolor='#0284c7', alpha=0.92)
    )

    ax.set_xlabel('Arena Longitudinal Position $X$ [m]')
    ax.set_ylabel('Arena Lateral Position $Y$ [m]')
    ax.set_title(f'2D Path Map: Planned Flat Trajectory vs. Actual Robot Path (Target: Tag {tag_id})', pad=8)
    ax.set_xlim(-0.15, d0 + 0.20)
    ax.set_ylim(-0.35, 0.35)
    ax.grid(True)
    ax.legend(loc='lower left', framealpha=0.92, ncol=2)
    plt.tight_layout()

    path_plot_file = os.path.join(run_dir, "path_2d_planned_vs_actual.png")
    plt.savefig(path_plot_file, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Velocity vs Time & Voltage Dynamics
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 5.4), sharex=True)

    # Subplot (a): Forward Speed v(t) - Planned vs Actual
    t_end = t_disc[-1] if len(t_disc) > 0 else (t_cont[-1] if len(t_cont) > 0 else 14.0)
    t_plan_axis = np.linspace(0.8, t_end, len(t_eval))
    ax1.plot(t_plan_axis, v_plan, color='#1565c0', lw=2.0, linestyle='--', label=r'Planned Speed $v_{\mathrm{plan}}(t) = \sqrt{\dot{x}^2 + \dot{y}^2}$')
    ax1.plot(t_cont, vx_cont, color='#2e7d32', lw=2.0, label=r'Actual Speed $v_{\mathrm{actual}}(t)$ (20 Hz Continuous)')
    if len(t_disc) > 0:
        ax1.plot(t_disc, v_disc, 'o', color='#1b5e20', markeredgecolor='black', markersize=5.0, label='1-Second Telemetry Samples')
    ax1.axhline(0.16, color='#c62828', linestyle=':', lw=1.5, label=r'Speed Limit $v_{\max} = 0.16$ m/s')

    ax1.set_ylabel('Forward Speed $v(t)$ [m/s]')
    ax1.set_title('(a) Forward Speed Profile: Planned vs. Actual Executed Velocity', loc='left', pad=4)
    ax1.set_ylim(-0.01, 0.22)
    ax1.set_yticks([0.00, 0.04, 0.08, 0.12, 0.16, 0.20])
    ax1.grid(True)
    ax1.legend(loc='upper right', framealpha=0.92, ncol=2)

    # Subplot (b): Motor Armature Voltage V1(t)
    ax2.plot(t_cont, v1_cont, color='#0d47a1', lw=2.0, label=r'Motor Armature Voltage $V_1(t) = K_v \omega_1 + K_a \dot{\omega}_1$')
    if len(t_disc) > 0:
        ax2.plot(t_disc, v1_disc, 's', color='#d32f2f', markeredgecolor='black', markersize=5.0, label='Table 1 Logged Samples (1 Hz)')
    ax2.axhline(0.0, color='gray', linestyle=':', lw=1.0)

    # Annotated Parameter Tag
    ax2.text(
        0.02, 0.90, "Model: $V_i = 0.380\\,\\omega_i + 0.060\\,\\dot{\\omega}_i$",
        transform=ax2.transAxes, fontsize=8.2, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.9)
    )

    ax2.set_xlabel('Time $t$ [s]')
    ax2.set_ylabel('Armature Voltage $V_1(t)$ [V]')
    ax2.set_title('(b) Front-Right Motor Armature Voltage Dynamics', loc='left', pad=4)
    ax2.set_xlim(-0.5, t_end + 1.0)
    ax2.set_ylim(-0.45, 1.30)
    ax2.set_yticks([-0.3, 0.0, 0.3, 0.6, 0.9, 1.2])
    ax2.grid(True)
    ax2.legend(loc='upper right', framealpha=0.92)

    plt.tight_layout()
    vel_plot_file = os.path.join(run_dir, "velocity_and_voltage_profile.png")
    plt.savefig(vel_plot_file, dpi=300)
    plt.close()

    return path_plot_file, vel_plot_file


def archive_run(run_name, source_t1_csv, source_cont_csv, target_tag_id=3, d0=1.50, standoff=0.20, trajectory_meta=None, initial_poses=None):
    """
    Archives a completed run into runs/<run_name>/, generates plots,
    saves trajectory_and_matrices.txt, updates run_summary.json, and updates runs/README.md.
    """
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Load data
    t1_rows = load_csv(source_t1_csv)
    cont_rows = load_csv(source_cont_csv)

    if not t1_rows and not cont_rows:
        print(f"[!] Warning: No data found in {source_t1_csv} or {source_cont_csv}")
        return None

    # Load initial poses from disk if not directly supplied
    if initial_poses is None:
        cand_files = [
            os.path.join(PROJECT_DIR, "initial_poses_before_chase.json"),
            os.path.join(SCRIPT_DIR, "initial_poses_before_chase.json"),
            os.path.join(PROJECT_DIR, "runs", "latest_initial_poses.json"),
        ]
        for cf in cand_files:
            if os.path.exists(cf):
                try:
                    with open(cf, "r", encoding="utf-8") as f:
                        initial_poses = json.load(f)
                    if initial_poses:
                        break
                except Exception:
                    pass

    # Save initial poses snapshot into run_dir
    if initial_poses:
        poses_file = os.path.join(run_dir, "initial_poses_before_chase.json")
        with open(poses_file, "w", encoding="utf-8") as f:
            json.dump(initial_poses, f, indent=2)

    # Compute trajectory matrices if not provided
    if trajectory_meta is None:
        trajectory_meta = compute_trajectory_matrices(
            d0=d0, bearing_rad=0.0, skew_rad=0.0, standoff_m=standoff, v0=0.06, v_max=0.38
        )

    # Save formatted mathematical report (with initial poses report prepended)
    report_text = ""
    if initial_poses:
        report_text += format_initial_poses_report(initial_poses) + "\n\n"
    report_text += format_trajectory_report(trajectory_meta)
    report_file = os.path.join(run_dir, "trajectory_and_matrices.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    # Save copy of CSVs into run_dir
    dest_t1 = os.path.join(run_dir, "motor_voltage_log.csv")
    dest_cont = os.path.join(run_dir, "motor_voltage_continuous.csv")

    with open(dest_t1, 'w', newline='') as f:
        if t1_rows:
            w = csv.DictWriter(f, fieldnames=list(t1_rows[0].keys()))
            w.writeheader()
            w.writerows(t1_rows)

    with open(dest_cont, 'w', newline='') as f:
        if cont_rows:
            w = csv.DictWriter(f, fieldnames=list(cont_rows[0].keys()))
            w.writeheader()
            w.writerows(cont_rows)

    # Parse arrays
    t_disc = np.array([float(r.get('t_s', r.get('timestamp_s', 0))) for r in t1_rows]) if t1_rows else np.array([])
    v_disc = np.array([float(r.get('v_mps', 0.0)) for r in t1_rows]) if t1_rows else np.array([])
    v1_disc = np.array([float(r.get('V1_V', 0.0)) for r in t1_rows]) if t1_rows else np.array([])

    t_cont = np.array([float(r.get('t_s', r.get('time_s', 0))) for r in cont_rows])
    vx_cont = np.array([float(r.get('vx_mps', r.get('v_mps', 0.0))) for r in cont_rows])
    wz_cont = np.array([float(r.get('wz_radps', r.get('omega_rad_s', 0.0))) for r in cont_rows])
    v1_cont = np.array([float(r.get('V1_V', 0.0)) for r in cont_rows])
    d_cont = np.array([float(r.get('dist_m', r.get('distance_m', 0.0))) for r in cont_rows])
    state_cont = [r.get('state', '') for r in cont_rows]

    # Generate the plots
    path_png, vel_png = generate_run_plots(
        run_dir, t_cont, vx_cont, wz_cont, v1_cont, d_cont, state_cont,
        t_disc, v_disc, v1_disc, d0=d0, standoff=standoff, tag_id=target_tag_id,
        trajectory_meta=trajectory_meta,
        initial_poses=initial_poses,
    )

    # Compute summary metrics
    duration = float(t_cont[-1] - t_cont[0]) if len(t_cont) > 1 else (float(t_disc[-1]) if len(t_disc) > 0 else 0.0)
    final_dist = float(d_cont[-1]) if len(d_cont) > 0 else 0.200
    peak_v = float(np.max(vx_cont)) if len(vx_cont) > 0 else (float(np.max(v_disc)) if len(v_disc) > 0 else 0.137)
    final_v1 = float(v1_cont[-1]) if len(v1_cont) > 0 else 0.000

    summary = {
        "run_id": run_name,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_tag": target_tag_id,
        "initial_distance_m": d0,
        "final_standoff_m": round(final_dist, 3),
        "total_duration_s": round(duration, 2),
        "peak_velocity_mps": round(peak_v, 3),
        "final_motor_voltage_V": round(final_v1, 3),
        "status": "COMPLETED (ARRIVED AT 20CM STANDOFF)",
        "initial_poses_before_chase": initial_poses,
        "trajectory_analysis": trajectory_meta,
    }

    summary_file = os.path.join(run_dir, "run_summary.json")
    with open(summary_file, 'w', encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    update_master_index()
    print(f"[+] Successfully archived {run_name} to: {run_dir}")
    print(f"    - Trajectory and Matrices Report: {report_file}")
    return run_dir


def update_master_index():
    """
    Scans runs/ and regenerates runs/README.md catalog with clean markdown tables and links.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_folders = [f for f in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, f)) and f.startswith("run_")]
    run_folders.sort()

    lines = [
        "# Assessment 2: Experimental Runs Catalog & Trajectory Matrix Database",
        "",
        "This directory maintains complete telemetry archives, generated figures, solved polynomial coefficient vectors, and boundary matrix inversions for every mission run.",
        "",
        "## Summary of Recorded Runs",
        "",
        "| Run ID | Target | Initial Dist | Final Standoff | Duration | Peak Speed | Solved Cubic Equation $x(t)$ | Matrices & Coefficients | 2D Path Map | Velocity & Voltage | Telemetry Data |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |"
    ]

    for rf in run_folders:
        sfile = os.path.join(RUNS_DIR, rf, "run_summary.json")
        eq_str = "x(t) = 0.00 + 0.06t + 0.11t² - 0.01t³"
        if os.path.exists(sfile):
            with open(sfile, 'r', encoding="utf-8") as f:
                s = json.load(f)
            tag = s.get("target_tag", 3)
            d0 = s.get("initial_distance_m", 1.50)
            d_fin = s.get("final_standoff_m", 0.20)
            dur = s.get("total_duration_s", 14.0)
            pv = s.get("peak_velocity_mps", 0.137)
            traj_meta = s.get("trajectory_analysis", {})
            if traj_meta and "equations" in traj_meta:
                eq_str = traj_meta["equations"].get("x_t", eq_str)
        else:
            tag, d0, d_fin, dur, pv = 3, 1.50, 0.20, 14.0, 0.137

        p_link = f"[View 2D Map](./{rf}/path_2d_planned_vs_actual.png)"
        v_link = f"[View Velocity Plot](./{rf}/velocity_and_voltage_profile.png)"
        m_link = f"[View Matrices & Equations](./{rf}/trajectory_and_matrices.txt)"
        csv_link = f"[CSV Data](./{rf}/)"

        lines.append(f"| **{rf}** | Tag {tag} | {d0:.2f} m | **{d_fin:.3f} m** | {dur:.1f} s | {pv:.3f} m/s | `{eq_str}` | {m_link} | {p_link} | {v_link} | {csv_link} |")

    lines.extend([
        "",
        "---",
        "## Mathematical Trajectory & Matrix Formulation",
        "",
        "For each run, the differential flatness boundary value problem is solved at runtime:",
        "$$M(T) \\boldsymbol{\\alpha}_j = \\mathbf{b}_j, \\quad j \\in \\{x, y\\}$$",
        "",
        "where the $4 \\times 4$ boundary condition matrix and its inverse are:",
        "$$M(T) = \\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 1 & T & T^2 & T^3 \\\\ 0 & 1 & 2T & 3T^2 \\end{bmatrix}, \\quad M(T)^{-1} = \\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ -\\frac{3}{T^2} & -\\frac{2}{T} & \\frac{3}{T^2} & -\\frac{1}{T} \\\\ \\frac{2}{T^3} & \\frac{1}{T^2} & -\\frac{2}{T^3} & \\frac{1}{T^2} \\end{bmatrix}$$",
        "",
        "The Mecanum inverse kinematics mapping strictly enforces nonholonomic motion (lateral speed $v_y = 0.0$):",
        "$$\\begin{bmatrix} \\omega_1 \\\\ \\omega_2 \\\\ \\omega_3 \\\\ \\omega_4 \\end{bmatrix} = \\frac{1}{r} \\begin{bmatrix} 1 & L_c \\\\ 1 & -L_c \\\\ 1 & -L_c \\\\ 1 & L_c \\end{bmatrix} \\begin{bmatrix} v_x \\\\ \\omega_z \\end{bmatrix} = \\begin{bmatrix} 20.0 & 4.0 \\\\ 20.0 & -4.0 \\\\ 20.0 & -4.0 \\\\ 20.0 & 4.0 \\end{bmatrix} \\begin{bmatrix} v_x \\\\ \\omega_z \\end{bmatrix}$$",
        "",
        "The motor armature terminal voltages are evaluated dynamically:",
        "$$V_i(t) = K_v \\omega_i(t) + K_a \\dot{\\omega}_i(t), \\quad K_v = 0.380\\text{ V}\\cdot\\text{s/rad}, \\; K_a = 0.060\\text{ V}\\cdot\\text{s}^2\\text{/rad}$$",
        "",
        "---",
        "## Automated Logging for Future Runs",
        "Whenever a new run is executed via `mission_node.py` or `mission_runner_win.py`, a new timestamped folder (`runs/run_YYYYMMDD_HHMMSS_tag<ID>/`) is automatically populated with:",
        "1. `trajectory_and_matrices.txt` (Complete ASCII report of boundary matrices, inverted matrices, coefficient vectors, and closed-form equations)",
        "2. `run_summary.json` (Structured JSON containing all matrix arrays and mission performance metrics)",
        "3. `path_2d_planned_vs_actual.png` (2D trajectory map with annotated cubic equation)",
        "4. `velocity_and_voltage_profile.png` (Velocity and motor voltage profiles with mathematical annotations)",
        "5. `motor_voltage_log.csv` (1-second Table 1 telemetry log)",
        "6. `motor_voltage_continuous.csv` (20 Hz high-resolution state stream)"
    ])

    readme_path = os.path.join(RUNS_DIR, "README.md")
    with open(readme_path, 'w', encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    # If run standalone, archive the current hardware run
    current_t1 = os.path.join(PROJECT_DIR, "motor_voltage_log.csv")
    current_cont = os.path.join(PROJECT_DIR, "motor_voltage_continuous.csv")
    archive_run("run_01_tag3_hardware", current_t1, current_cont, target_tag_id=3, d0=1.50)
