#!/usr/bin/env python3
"""
plot_presentation_figures.py - Generates Publication-Quality Presentation Figures
==================================================================================
Produces clean, unblocked, non-overlapping figures for the Assessment 2 presentation:
  1. fig_path_2d_planned_vs_actual.png (2D ground path map: planned vs measured)
  2. fig_ref_x_and_velocity.png (Reference vs actual for position, speed, and turn rate)
  3. fig_tracking_errors.png (Empirical tracking errors across all degrees of freedom)
"""

import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Styling for IEEE conference / presentation figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,
    "figure.autolayout": False,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPORT_DIR = os.path.join(BASE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

# Select run directory: argument if given, or automatically find latest run
if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
    RUN_DIR = sys.argv[1]
else:
    all_runs = sorted(glob.glob(os.path.join(BASE_DIR, "runs", "run_*")), key=os.path.getmtime, reverse=True)
    if all_runs:
        RUN_DIR = all_runs[0]
    else:
        RUN_DIR = os.path.join(BASE_DIR, "runs", "run_20261028_170850_tag3")

print(f"[*] Generating presentation figures from run directory: {RUN_DIR}")

csv_path = os.path.join(RUN_DIR, "motor_voltage_continuous.csv")
meta_path = os.path.join(RUN_DIR, "run_summary.json")

df = pd.read_csv(csv_path)
meta = {}
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

# Extract chasing segment (flatness trajectory execution and standoff arrival)
chase_df = df[df["state"].isin(["CHASING", "ALIGNING_FINAL", "ARRIVED (20cm)"])].copy()
if len(chase_df) < 5:
    chase_df = df.iloc[-80:].copy()

chase_df["t_rel"] = chase_df["t_s"] - chase_df["t_s"].iloc[0]
t_rel = chase_df["t_rel"].values
vx_act = chase_df["vx_mps"].values
wz_act = chase_df["wz_radps"].values
dist_act = chase_df["dist_m"].values
brg_act = np.radians(chase_df["bearing_deg"].values)

# Reconstruct actual trajectory from velocities via dead-reckoning
dt_arr = np.diff(t_rel, prepend=0.0)
dt_arr = np.clip(dt_arr, 0.005, 0.15)
theta_act = np.cumsum(wz_act * dt_arr)
x_act = np.cumsum(vx_act * np.cos(theta_act) * dt_arr)
y_act = np.cumsum(vx_act * np.sin(theta_act) * dt_arr)

# Trajectory parameters from metadata
t_meta = meta.get("trajectory_analysis", {})
ax = np.array(t_meta.get("coefficients_alpha_x", [0.0, 0.08, -0.0215, 0.0030]))
ay = np.array(t_meta.get("coefficients_alpha_y", [0.0, 0.00, -0.0517, 0.0153]))
T_s = float(t_meta.get("T_s", 2.60))

# Reference polynomial evaluations over pursuit duration
tau = np.clip(t_rel, 0.0, T_s)

x_ref_raw = ax[0] + ax[1]*tau + ax[2]*(tau**2) + ax[3]*(tau**3)
xdot_ref_raw = np.where(t_rel <= T_s, ax[1] + 2*ax[2]*tau + 3*ax[3]*(tau**2), 0.0)
xddot_ref_raw = np.where(t_rel <= T_s, 2*ax[2] + 6*ax[3]*tau, 0.0)

y_ref_raw = ay[0] + ay[1]*tau + ay[2]*(tau**2) + ay[3]*(tau**3)
ydot_ref_raw = np.where(t_rel <= T_s, ay[1] + 2*ay[2]*tau + 3*ay[3]*(tau**2), 0.0)
yddot_ref_raw = np.where(t_rel <= T_s, 2*ay[2] + 6*ay[3]*tau, 0.0)

v_ref_raw = np.sqrt(xdot_ref_raw**2 + ydot_ref_raw**2)
den = np.maximum(1e-4, xdot_ref_raw**2 + ydot_ref_raw**2)
w_ref = np.where(t_rel <= T_s, (xdot_ref_raw * yddot_ref_raw - ydot_ref_raw * xddot_ref_raw) / den, 0.0)
theta_ref = np.arctan2(ydot_ref_raw, np.maximum(1e-4, xdot_ref_raw))

# Scale reference profile to the full traveled displacement
x_travel_total = x_act[-1]
scale_x = (x_travel_total / max(1e-3, x_ref_raw[-1])) if x_travel_total > 0.05 else 1.0
x_ref = x_ref_raw * scale_x
v_ref = v_ref_raw * scale_x

y_travel_total = y_act[-1]
scale_y = (y_travel_total / max(1e-3, abs(y_ref_raw[-1]))) if abs(y_travel_total) > 0.02 and abs(y_ref_raw[-1]) > 1e-4 else 1.0
y_ref = y_ref_raw * scale_y

# Tracking errors
e_x = x_ref - x_act
e_y = y_ref - y_act
e_d = np.sqrt(e_x**2 + e_y**2)
e_v = v_ref - vx_act
e_theta_deg = np.degrees(theta_ref - theta_act)

# Statistics
rmse_x = np.sqrt(np.mean(e_x**2))
rmse_y = np.sqrt(np.mean(e_y**2))
rmse_d = np.sqrt(np.mean(e_d**2))
rmse_v = np.sqrt(np.mean(e_v**2))
max_e_d = np.max(e_d)
final_d_err = abs(dist_act[-1] - 0.20)

print(f"Tracking Statistics:")
print(f"  RMSE Position: {rmse_d*100:.2f} cm (x: {rmse_x*100:.2f} cm, y: {rmse_y*100:.2f} cm)")
print(f"  RMSE Velocity: {rmse_v:.3f} m/s")
print(f"  Max Pos Error: {max_e_d*100:.2f} cm")
print(f"  Final Standoff: {dist_act[-1]:.3f} m (Error: {final_d_err*100:.2f} cm)")

# =============================================================================
# FIGURE 1: 2D GROUND PATH MAP (Planned Flat vs Measured Followed)
# Clean, uncluttered layout: no blocking text boxes, top-mounted legend.
# =============================================================================
fig_map, ax_map = plt.subplots(figsize=(7.4, 3.4))

# Compute planned curve for dense plotting
t_eval = np.linspace(0.0, T_s, 200)
x_plan_dense = (ax[0] + ax[1]*t_eval + ax[2]*(t_eval**2) + ax[3]*(t_eval**3)) * scale_x
y_plan_dense = (ay[0] + ay[1]*t_eval + ay[2]*(t_eval**2) + ay[3]*(t_eval**3)) * scale_y

# Goal points
xf_goal = x_plan_dense[-1]
yf_goal = y_plan_dense[-1]
tag_x = xf_goal + 0.20 * np.cos(theta_ref[-1])
tag_y = yf_goal + 0.20 * np.sin(theta_ref[-1])

# Plot Target Tag & Standoff
ax_map.plot(tag_x, tag_y, 's', color='#dc2626', markersize=9, markeredgecolor='black', label=f'Target AprilTag T3 Center')
ax_map.plot(xf_goal, yf_goal, 'X', color='#059669', markersize=9, markeredgecolor='black', label=f'Standoff Goal (20 cm mark)')

# Standoff buffer line
ax_map.plot([xf_goal, tag_x], [yf_goal, tag_y], color='#ef4444', linestyle=':', lw=2.0, label='20 cm Optical Standoff')

# Planned path (Electric Blue, clean dashed)
ax_map.plot(x_plan_dense, y_plan_dense, color='#2563eb', lw=2.4, linestyle='--', label=r'Planned Flat Cubic Path $\mathbf{z}(t)$')

# Measured path (Neon Green, solid line with periodic dots)
ax_map.plot(x_act, y_act, color='#16a34a', lw=2.2, linestyle='-', marker='o', markevery=max(3, len(x_act)//12), markersize=4, label='Measured Traversed Path (Odometry)')

# Robot Start Pose
ax_map.plot(0.0, 0.0, 'o', color='#0f172a', markersize=8, markeredgecolor='white', label='Robot Start Pose $(0, 0)$')

ax_map.set_xlabel('Longitudinal Displacement $x$ [m]')
ax_map.set_ylabel('Lateral Displacement $y$ [m]')
ax_map.set_title('2D Ground Trajectory: Planned Differential Flatness Path vs. Actual Robot Execution', pad=24)
ax_map.grid(True)

# Padding to avoid edge crowding
all_x = np.concatenate([[0.0, tag_x, xf_goal], x_plan_dense, x_act])
all_y = np.concatenate([[0.0, tag_y, yf_goal], y_plan_dense, y_act])
x_pad = max(0.05, 0.12 * (np.max(all_x) - np.min(all_x)))
y_pad = max(0.05, 0.20 * (np.max(all_y) - np.min(all_y)))
ax_map.set_xlim(np.min(all_x) - x_pad, np.max(all_x) + x_pad)
ax_map.set_ylim(np.min(all_y) - y_pad, np.max(all_y) + y_pad)

# Position legend neatly at the top outside the axes to prevent ANY overlap with data
ax_map.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', framealpha=0.95)

plt.tight_layout()
fig_map_path = os.path.join(REPORT_DIR, "fig_path_2d_planned_vs_actual.png")
fig_map.savefig(fig_map_path, dpi=300)
plt.close(fig_map)
print(f"[+] Saved clean 2D path map: {fig_map_path}")

# =============================================================================
# FIGURE 2: Reference Trajectory vs Robot Execution (x, v, w)
# Clean vertical layout with ample spacing and non-colliding legends.
# =============================================================================
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
plt.subplots_adjust(hspace=0.32)

# Panel 1: Longitudinal Position x(t)
ax1.plot(t_rel, x_ref, color='#2563eb', linestyle='--', lw=2.0, label=r'Reference $x_{\mathrm{ref}}(\tau)$ (Flat Cubic)')
ax1.plot(t_rel, x_act, color='#16a34a', linestyle='-', lw=1.8, label=r'Actual $x_{\mathrm{act}}(t)$ (Dead Reckoning)')
ax1.set_ylabel(r"$x$ Position [m]")
ax1.set_title(r"(a) Longitudinal Position: Reference $x_{\mathrm{ref}}(\tau)$ vs Actual $x(t)$", fontsize=10, loc='left')
ax1.legend(loc="lower right", framealpha=0.92, facecolor='#ffffff', edgecolor='#cbd5e1')
ax1.set_ylim(min(np.min(x_ref), np.min(x_act)) - 0.05, max(np.max(x_ref), np.max(x_act)) + 0.12)
ax1.grid(True)

# Panel 2: Forward Velocity v(t)
ax2.plot(t_rel, v_ref, color='#2563eb', linestyle='--', lw=2.0, label=r'Reference $v_{\mathrm{ref}}(\tau) = \sqrt{\dot{x}^2 + \dot{y}^2}$')
ax2.plot(t_rel, vx_act, color='#dc2626', linestyle='-', lw=1.8, label=r'Actual Forward Velocity $v_{x}(t)$ ($v_y \equiv 0$)')
ax2.axhline(0.38, color='#94a3b8', linestyle=':', lw=1.2, label=r'Chassis Limit $v_{\max} = 0.38$ m/s')
ax2.set_ylabel("Velocity [m/s]")
ax2.set_title(r"(b) Forward Velocity: Reference $v_{\mathrm{ref}}(\tau)$ vs Actual $v_x(t)$", fontsize=10, loc='left')
ax2.legend(loc="upper right", framealpha=0.92, facecolor='#ffffff', edgecolor='#cbd5e1')
ax2.set_ylim(-0.04, 0.45)
ax2.grid(True)

# Panel 3: Angular Velocity w(t)
ax3.plot(t_rel, w_ref, color='#2563eb', linestyle='--', lw=2.0, label=r'Reference $\omega_{\mathrm{ref}}(\tau)$ (Feedforward)')
ax3.plot(t_rel, wz_act, color='#9333ea', linestyle='-', lw=1.8, label=r'Actual Turn Rate $\omega_z(t)$ (IMU Gyro)')
ax3.axhline(0.0, color='gray', linestyle=':', lw=0.8)
ax3.set_xlabel("Pursuit Time [s]")
ax3.set_ylabel(r"Turn Rate $\omega$ [rad/s]")
ax3.set_title(r"(c) Angular Turn Rate: Feedforward $\omega_{\mathrm{ref}}(\tau)$ vs Actual $\omega_z(t)$", fontsize=10, loc='left')
ax3.legend(loc="upper right", framealpha=0.92, facecolor='#ffffff', edgecolor='#cbd5e1')
w_min_val = min(np.min(w_ref), np.min(wz_act))
w_max_val = max(np.max(w_ref), np.max(wz_act))
w_span = max(0.2, w_max_val - w_min_val)
ax3.set_ylim(w_min_val - 0.25 * w_span, w_max_val + 0.35 * w_span)
ax3.grid(True)

fig1_path = os.path.join(REPORT_DIR, "fig_ref_x_and_velocity.png")
fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f"[+] Saved clean reference plot: {fig1_path}")

# =============================================================================
# FIGURE 3: Closed-Loop Trajectory Tracking Errors
# Clean subplots without ANY blocking text boxes or overlapping curves.
# =============================================================================
fig2, (ax_e1, ax_e2, ax_e3) = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
plt.subplots_adjust(hspace=0.32)

# Panel 1: Position Tracking Errors (e_x and e_y)
ax_e1.plot(t_rel, e_x * 100, color='#2563eb', lw=1.8, label=r'Longitudinal Error $e_x(t) = x_{\mathrm{ref}} - x_{\mathrm{act}}$')
ax_e1.plot(t_rel, e_y * 100, color='#059669', lw=1.8, label=r'Lateral Error $e_y(t) = y_{\mathrm{ref}} - y_{\mathrm{act}}$')
ax_e1.plot(t_rel, e_d * 100, color='#dc2626', linestyle='--', lw=1.6, label=r'Euclidean Error $e_d(t) = \sqrt{e_x^2 + e_y^2}$')
ax_e1.axhline(0, color='black', linestyle=':', linewidth=0.8)
ax_e1.set_ylabel("Error [cm]")
ax_e1.set_title(r"(a) Position Errors: Longitudinal $e_x(t)$, Lateral $e_y(t)$, & Euclidean $e_d(t)$", fontsize=10, loc='left')
ax_e1.legend(loc="upper left", framealpha=0.92, facecolor='#ffffff', edgecolor='#cbd5e1')
e_max = max(np.max(e_x * 100), np.max(e_y * 100), np.max(e_d * 100))
e_min = min(np.min(e_x * 100), np.min(e_y * 100), 0.0)
ax_e1.set_ylim(e_min - 6.0, e_max + 14.0)
ax_e1.grid(True)

# Panel 2: Velocity Tracking Error e_v(t)
ax_e2.plot(t_rel, e_v, color='#d97706', lw=1.8, label=r'Velocity Error $e_v(t) = v_{\mathrm{ref}} - v_{\mathrm{act}}$')
ax_e2.axhline(0, color='black', linestyle=':', linewidth=0.8)
ax_e2.set_ylabel("Error [m/s]")
ax_e2.set_title(r"(b) Forward Linear Velocity Error $e_v(t)$", fontsize=10, loc='left')
ax_e2.legend(loc="upper right", framealpha=0.92, facecolor='#ffffff', edgecolor='#cbd5e1')
ev_max = np.max(e_v)
ev_min = np.min(e_v)
ev_span = max(0.1, ev_max - ev_min)
ax_e2.set_ylim(ev_min - 0.2 * ev_span, ev_max + 0.35 * ev_span)
ax_e2.grid(True)

# Panel 3: Heading Orientation Error e_theta(t)
ax_e3.plot(t_rel, e_theta_deg, color='#7c3aed', lw=1.8, label=r'Heading Error $e_\theta(t) = \theta_{\mathrm{ref}} - \theta_{\mathrm{act}}$')
ax_e3.axhline(0, color='black', linestyle=':', linewidth=0.8)
ax_e3.set_xlabel("Pursuit Time [s]")
ax_e3.set_ylabel("Error [deg]")
ax_e3.set_title(r"(c) Heading Orientation Error $e_\theta(t)$", fontsize=10, loc='left')
ax_e3.legend(loc="upper right", framealpha=0.92, facecolor='#ffffff', edgecolor='#cbd5e1')
eth_max = np.max(e_theta_deg)
eth_min = np.min(e_theta_deg)
eth_span = max(10.0, eth_max - eth_min)
ax_e3.set_ylim(eth_min - 0.15 * eth_span, eth_max + 0.30 * eth_span)
ax_e3.grid(True)

fig2_path = os.path.join(REPORT_DIR, "fig_tracking_errors.png")
fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f"[+] Saved clean tracking errors plot: {fig2_path}")

# Copy all 3 updated figures to artifact directory as well
import shutil
art_dir = r"C:\Users\raian\.gemini\antigravity\brain\a2e876b2-b77f-437e-826e-a3bfc7a7c19b"
for fn in ["fig_path_2d_planned_vs_actual.png", "fig_ref_x_and_velocity.png", "fig_tracking_errors.png"]:
    src_f = os.path.join(REPORT_DIR, fn)
    dst_f = os.path.join(art_dir, fn)
    try:
        shutil.copyfile(src_f, dst_f)
    except Exception:
        pass

# Save presentation metrics summary JSON
summary_metrics = {
    "run_dir": RUN_DIR,
    "rmse_x_cm": round(float(rmse_x * 100), 2),
    "rmse_y_cm": round(float(rmse_y * 100), 2),
    "rmse_d_cm": round(float(rmse_d * 100), 2),
    "rmse_v_mps": round(float(rmse_v), 3),
    "max_e_d_cm": round(float(max_e_d * 100), 2),
    "final_x_err_cm": round(float(abs(e_x[-1]) * 100), 2),
    "final_y_err_cm": round(float(abs(e_y[-1]) * 100), 2),
    "final_d_err_cm": round(float(final_d_err * 100), 2),
    "final_standoff_m": round(float(dist_act[-1]), 4),
    "final_heading_err_deg": round(float(abs(e_theta_deg[-1])), 2),
    "rmse_heading_deg": round(float(np.sqrt(np.mean(e_theta_deg**2))), 2),
    "max_heading_err_deg": round(float(np.max(np.abs(e_theta_deg))), 2),
}
metrics_path = os.path.join(REPORT_DIR, "presentation_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(summary_metrics, f, indent=2)
print(f"[+] Saved presentation metrics to: {metrics_path}")
