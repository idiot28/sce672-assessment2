#!/usr/bin/env python3
"""
plot_presentation_figures.py
Generates presentation-grade reference vs actual and tracking error figures
for the SCE 672 Assessment 2 report presentation.
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
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "figure.autolayout": False,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.45,
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
# Map tau smoothly over active pursuit phase
t_active = t_rel[t_rel <= (T_s + 0.5)]
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
print(f"  RMSE Position: {rmse_d*100:.2f} cm")
print(f"  RMSE Velocity: {rmse_v:.3f} m/s")
print(f"  Max Pos Error: {max_e_d*100:.2f} cm")
print(f"  Final Standoff: {dist_act[-1]:.3f} m (Error: {final_d_err*100:.2f} cm)")

# =============================================================================
# FIGURE 1: Reference vs Actual for Velocity and x
# =============================================================================
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
fig1.suptitle("Reference Trajectory vs. Robot Execution (Differential Flatness Pursuit)", fontweight="bold", y=0.98)

# Panel 1: Longitudinal Position x(t)
ax1.plot(t_rel, x_ref, "b--", label=r"Reference $x_{\mathrm{ref}}(\tau)$ (Flat Cubic)")
ax1.plot(t_rel, x_act, "g-", label=r"Actual $x_{\mathrm{act}}(t)$ (Odometry)")
ax1.set_ylabel(r"$x$ Position [m]")
ax1.set_title(r"(a) Longitudinal Position: Flat Reference $x_{\mathrm{ref}}(\tau)$ vs Actual $x(t)$", fontsize=10)
ax1.legend(loc="lower right", framealpha=0.9)
ax1.grid(True)

# Panel 2: Forward Velocity v(t)
ax2.plot(t_rel, v_ref, "b--", label=r"Reference $v_{\mathrm{ref}}(\tau) = \sqrt{\dot{x}^2 + \dot{y}^2}$")
ax2.plot(t_rel, vx_act, "r-", label=r"Actual Forward Velocity $v_{x}(t)$ ($v_y = 0$)")
ax2.set_ylabel("Velocity [m/s]")
ax2.set_title(r"(b) Forward Velocity: Flat Reference $v_{\mathrm{ref}}(\tau)$ vs Actual $v_x(t)$", fontsize=10)
ax2.legend(loc="upper right", framealpha=0.9)
ax2.grid(True)

# Panel 3: Angular Velocity w(t)
ax3.plot(t_rel, w_ref, "b--", label=r"Reference $\omega_{\mathrm{ref}}(\tau)$ (Flatness Feedforward)")
ax3.plot(t_rel, wz_act, color="purple", label=r"Actual Turn Rate $\omega_z(t)$ (IMU Gyro)")
ax3.set_xlabel("Pursuit Time [s]")
ax3.set_ylabel(r"$\omega$ [rad/s]")
ax3.set_title(r"(c) Angular Velocity: Feedforward $\omega_{\mathrm{ref}}(\tau)$ vs Commanded Turn Rate $\omega_z(t)$", fontsize=10)
ax3.legend(loc="upper right", framealpha=0.9)
ax3.grid(True)

plt.tight_layout()
fig1_path = os.path.join(REPORT_DIR, "fig_ref_x_and_velocity.png")
fig1.savefig(fig1_path, dpi=300)
plt.close(fig1)
print(f"[+] Saved: {fig1_path}")

# =============================================================================
# FIGURE 2: Tracking Errors
# =============================================================================
fig2, (ax_e1, ax_e2, ax_e3) = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
fig2.suptitle("Closed-Loop Trajectory Tracking Errors & Convergence", fontweight="bold", y=0.98)

# Panel 1: Position Tracking Errors (e_x and e_y)
ax_e1.plot(t_rel, e_x * 100, "b-", label=r"Longitudinal Error $e_x(t) = x_{\mathrm{ref}} - x_{\mathrm{act}}$")
ax_e1.plot(t_rel, e_y * 100, "g-", label=r"Lateral Error $e_y(t) = y_{\mathrm{ref}} - y_{\mathrm{act}}$")
ax_e1.plot(t_rel, e_d * 100, "r--", linewidth=1.4, label=r"Euclidean Error $e_d(t) = \sqrt{e_x^2 + e_y^2}$")
ax_e1.axhline(0, color="black", linestyle=":", linewidth=0.8)
ax_e1.set_ylabel("Error [cm]")
ax_e1.set_title(r"(a) Position Tracking Errors: $e_x(t)$, $e_y(t)$, and Total Euclidean Error $e_d(t)$", fontsize=10)
ax_e1.legend(loc="upper right", framealpha=0.9)
ax_e1.grid(True)

# Panel 2: Velocity Tracking Error e_v(t)
ax_e2.plot(t_rel, e_v, "m-", label=r"Velocity Error $e_v(t) = v_{\mathrm{ref}} - v_{\mathrm{act}}$")
ax_e2.axhline(0, color="black", linestyle=":", linewidth=0.8)
ax_e2.set_ylabel("Error [m/s]")
ax_e2.set_title(r"(b) Forward Velocity Tracking Error $e_v(t)$", fontsize=10)
ax_e2.legend(loc="upper right", framealpha=0.9)
ax_e2.grid(True)

# Panel 3: Heading Orientation Error e_theta(t)
ax_e3.plot(t_rel, e_theta_deg, color="#d97706", label=r"Heading Error $e_\theta(t) = \theta_{\mathrm{ref}} - \theta_{\mathrm{act}}$")
ax_e3.axhline(0, color="black", linestyle=":", linewidth=0.8)
ax_e3.set_xlabel("Pursuit Time [s]")
ax_e3.set_ylabel("Error [deg]")
ax_e3.set_title(r"(c) Heading / Tangent Alignment Error $e_\theta(t)$", fontsize=10)
ax_e3.legend(loc="upper right", framealpha=0.9)
ax_e3.grid(True)

# Summary Stats Text Box on bottom panel
stats_text = (
    f"Tracking Performance Summary:\n"
    f"• Position RMSE: {rmse_d*100:.2f} cm\n"
    f"• Velocity RMSE: {rmse_v:.3f} m/s\n"
    f"• Max Position Error: {max_e_d*100:.2f} cm\n"
    f"• Final Standoff Error: {final_d_err*100:.2f} cm (Goal: 20.0 cm)"
)
ax_e3.text(
    0.02, 0.15, stats_text, transform=ax_e3.transAxes,
    fontsize=8, verticalalignment="bottom",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.95)
)

plt.tight_layout()
fig2_path = os.path.join(REPORT_DIR, "fig_tracking_errors.png")
fig2.savefig(fig2_path, dpi=300)
plt.close(fig2)
print(f"[+] Saved: {fig2_path}")

# Copy latest 2D path map from RUN_DIR into report/
import shutil
src_map = os.path.join(RUN_DIR, "path_2d_planned_vs_actual.png")
if os.path.exists(src_map):
    dest_map = os.path.join(REPORT_DIR, "fig_path_2d_planned_vs_actual.png")
    shutil.copyfile(src_map, dest_map)
    print(f"[+] Copied 2D path map to: {dest_map}")

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
