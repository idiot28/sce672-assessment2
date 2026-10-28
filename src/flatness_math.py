#!/usr/bin/env python3
"""
flatness_math.py - Differential Flatness & Motor Voltage Equations (Assessment 2 - v4)
======================================================================================
Directly implements all equations from the SCE 672 Assessment 2 Instructions:
  - Eq. (1): Differential Flatness Inversion (v(t), omega(t))
  - Eq. (2): Cubic Polynomial Boundary Value Matrix Solver (solve_flat_polynomial)
  - Eq. (3): Dynamic Time Horizon Scaling (compute_time_horizon)
  - Eq. (4): Nonholonomic Wheel Kinematics (no strafe vy=0; only vx and wz)
  - Eq. (5): Motor Armature Voltage Model (Back-EMF Kv + Inertia Ka)
"""

import math
import numpy as np

# Physical Constants from Instructions (Problem 3)
WHEEL_RADIUS_R = 0.05      # r  = 0.05 m
CHASSIS_RADIUS_LC = 0.20   # Lc = 0.20 m
KV_BACK_EMF = 0.38         # Kv = 0.38 V*s/rad
KA_INERTIA = 0.06          # Ka = 0.06 V*s^2/rad


def wrap_angle(angle_rad: float) -> float:
    """Wraps any angle in radians to [-pi, +pi]."""
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


# =============================================================================
# Problem 2(ii): Cubic Polynomial Matrix Solution (Eq. 2)
# =============================================================================
def solve_flat_polynomial(z0: float, zdot0: float, zf: float, zdotf: float, T: float) -> np.ndarray:
    """
    Solves the 4x4 boundary value matrix equation (Eq. 2) using numpy.linalg.solve:
        z_j(t) = alpha_0 + alpha_1*t + alpha_2*t^2 + alpha_3*t^3
    """
    T = max(float(T), 0.1)
    M = np.array([
        [1.0, 0.0, 0.0,     0.0],
        [0.0, 1.0, 0.0,     0.0],
        [1.0, T,   T**2,    T**3],
        [0.0, 1.0, 2.0 * T, 3.0 * (T**2)],
    ], dtype=np.float64)
    b = np.array([z0, zdot0, zf, zdotf], dtype=np.float64)
    return np.linalg.solve(M, b)


# =============================================================================
# Problem 2(iii): Dynamic Time Horizon Scaling (Eq. 3)
# =============================================================================
def compute_time_horizon(d0: float, v_max: float = 0.38, min_T: float = 2.6) -> float:
    """
    Computes trajectory duration T dynamically from initial tag distance d0 (Eq. 3):
        T = max(min_T, 1.6 * (d0 - 0.20) / v_max)
    With fast v_max = 0.38 m/s, T is brisk (~2.6s - 4.5s).
    """
    travel = max(0.08, float(d0) - 0.20)
    T = (1.6 * travel) / max(float(v_max), 0.10)
    return max(float(min_T), float(T))


# =============================================================================
# Problem 2(i): Differential Flatness Inversion (Eq. 1)
# =============================================================================
def evaluate_flatness(alpha_x: np.ndarray, alpha_y: np.ndarray, t: float, T: float):
    """
    Evaluates the 2D flat cubic curve z(t) = [x(t), y(t)]^T and recovers
    forward speed v(t) and angular turn rate omega(t) via Eq. (1):
        v(t)     = sqrt(xdot^2 + ydot^2)
        omega(t) = (xdot * yddot - ydot * xddot) / (xdot^2 + ydot^2)
    """
    tau = float(np.clip(t, 0.0, T))
    tau2 = tau * tau
    tau3 = tau2 * tau

    x = alpha_x[0] + alpha_x[1] * tau + alpha_x[2] * tau2 + alpha_x[3] * tau3
    xdot = alpha_x[1] + 2.0 * alpha_x[2] * tau + 3.0 * alpha_x[3] * tau2
    xddot = 2.0 * alpha_x[2] + 6.0 * alpha_x[3] * tau

    y = alpha_y[0] + alpha_y[1] * tau + alpha_y[2] * tau2 + alpha_y[3] * tau3
    ydot = alpha_y[1] + 2.0 * alpha_y[2] * tau + 3.0 * alpha_y[3] * tau2
    yddot = 2.0 * alpha_y[2] + 6.0 * alpha_y[3] * tau

    speed_sq = xdot * xdot + ydot * ydot
    v = math.sqrt(speed_sq)
    omega = (xdot * yddot - ydot * xddot) / max(speed_sq, 1e-6)

    return float(v), float(omega), float(x), float(y), float(xdot), float(ydot)


def plan_aligning_cubic_path(
    d0: float,
    bearing_rad: float,
    skew_rad: float,
    standoff_m: float = 0.20,
    v0: float = 0.16,
    v_max: float = 0.38,
):
    """
    Plans a 2D flat cubic trajectory z(t) = [x(t), y(t)]^T in the robot's initial chase frame
    that moves from (0, 0) to the 20 cm standoff point directly in front of the tag,
    curving smoothly so the robot aligns straight/parallel with the tag at arrival.
    """
    T = compute_time_horizon(d0, v_max=v_max)

    # Tag position in robot frame at start of chase
    x_tag = d0 * math.cos(bearing_rad)
    y_tag = d0 * math.sin(bearing_rad)

    # Desired final heading to face the tag straight and eliminate horizontal skew
    goal_heading = wrap_angle(bearing_rad - 0.75 * skew_rad)

    # Standoff point 20 cm in front of the tag along its normal axis
    xf = max(0.08, x_tag - standoff_m * math.cos(goal_heading))
    yf = y_tag - standoff_m * math.sin(goal_heading)

    # Boundary conditions:
    # Start (t=0): x(0)=0, y(0)=0, moving straight forward in robot frame: xdot(0)=v0, ydot(0)=0
    # End   (t=T): x(T)=xf, y(T)=yf, decelerating to a halt aligned with goal_heading (xdot(T)=0, ydot(T)=0)
    alpha_x = solve_flat_polynomial(0.0, v0, xf, 0.0, T)
    alpha_y = solve_flat_polynomial(0.0, 0.0, yf, 0.0, T)
    return alpha_x, alpha_y, T, xf, yf


# =============================================================================
# Problem 3(i): Nonholonomic Kinematics to Wheel Rates (Eq. 4 - No Strafe!)
# =============================================================================
def wheel_kinematics(
    vx: float,
    wz: float,
    r: float = WHEEL_RADIUS_R,
    Lc: float = CHASSIS_RADIUS_LC,
) -> np.ndarray:
    """
    Maps forward/backward speed vx and turn rate wz to the 4 wheel angular speeds (Eq. 4).
    Strictly nonholonomic: lateral velocity vy is ALWAYS 0.0 (no strafing).
        w1 = w_FR = (vx + Lc * wz) / r
        w2 = w_FL = (vx - Lc * wz) / r
        w3 = w_RL = w2
        w4 = w_RR = w1
    """
    w1 = (float(vx) + Lc * float(wz)) / r
    w2 = (float(vx) - Lc * float(wz)) / r
    w3 = w2
    w4 = w1
    return np.array([w1, w2, w3, w4], dtype=np.float64)


# =============================================================================
# Problem 3(ii): Motor Armature Voltage Model (Eq. 5)
# =============================================================================
def motor_voltages(
    w_curr: np.ndarray,
    w_prev: np.ndarray,
    dt: float,
    halted: bool = False,
    Kv: float = KV_BACK_EMF,
    Ka: float = KA_INERTIA,
) -> np.ndarray:
    """
    Computes terminal voltages V1..V4 of all four chassis motors (Eq. 5):
        V_i(t) = Kv * w_i(t) + Ka * (w_i(t) - w_i(t - dt)) / dt
    Drops to 0.000 V when halted at the 20 cm standoff mark.
    """
    if halted:
        return np.zeros(4, dtype=np.float64)
    dt_safe = max(float(dt), 0.02)
    w_dot = np.clip((w_curr - w_prev) / dt_safe, -15.0, 15.0)
    return Kv * w_curr + Ka * w_dot
