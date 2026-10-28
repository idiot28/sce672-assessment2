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


def compute_boundary_matrix(T: float) -> np.ndarray:
    """Returns the 4x4 boundary condition matrix M(T) for cubic polynomial trajectory."""
    T = max(float(T), 0.1)
    return np.array([
        [1.0, 0.0, 0.0,     0.0],
        [0.0, 1.0, 0.0,     0.0],
        [1.0, T,   T**2,    T**3],
        [0.0, 1.0, 2.0 * T, 3.0 * (T**2)],
    ], dtype=np.float64)


def compute_boundary_matrix_inv(T: float) -> np.ndarray:
    """Returns the exact analytical 4x4 inverse boundary matrix M(T)^(-1)."""
    T = max(float(T), 0.1)
    T2 = T * T
    T3 = T2 * T
    return np.array([
        [1.0,        0.0,       0.0,        0.0],
        [0.0,        1.0,       0.0,        0.0],
        [-3.0 / T2, -2.0 / T,   3.0 / T2,  -1.0 / T],
        [ 2.0 / T3,  1.0 / T2, -2.0 / T3,   1.0 / T2],
    ], dtype=np.float64)


def get_mecanum_jacobian(r: float = WHEEL_RADIUS_R, Lc: float = CHASSIS_RADIUS_LC) -> np.ndarray:
    """
    Returns the 4x2 Mecanum Inverse Kinematics Jacobian matrix J:
        [w1, w2, w3, w4]^T = J * [vx, wz]^T
        J = 1/r * [[1,  Lc],
                   [1, -Lc],
                   [1, -Lc],
                   [1,  Lc]]
    """
    return (1.0 / float(r)) * np.array([
        [1.0,  float(Lc)],
        [1.0, -float(Lc)],
        [1.0, -float(Lc)],
        [1.0,  float(Lc)],
    ], dtype=np.float64)


def compute_trajectory_matrices(
    d0: float,
    bearing_rad: float,
    skew_rad: float,
    standoff_m: float = 0.20,
    v0: float = 0.16,
    v_max: float = 0.38,
    vf: float = 0.05,
) -> dict:
    """
    Computes all mathematical matrices, boundary vectors, solved polynomial
    coefficients, and closed-form equations for Assessment 2 differential flatness.
    Terminal velocity is aligned with the tag normal vector:
        zdotf_x = vf * cos(goal_heading),  zdotf_y = vf * sin(goal_heading)
    Guarantees the robot arrives directly perpendicular to the tag face (0 deg skew).
    """
    T = compute_time_horizon(d0, v_max=v_max)
    x_tag = d0 * math.cos(bearing_rad)
    y_tag = d0 * math.sin(bearing_rad)
    goal_heading = wrap_angle(bearing_rad - 0.25 * skew_rad) if abs(skew_rad) > 0.04 else bearing_rad
    xf = max(0.08, x_tag - standoff_m * math.cos(goal_heading))
    yf = y_tag - standoff_m * math.sin(goal_heading)

    face_tag_heading = math.atan2(y_tag - yf, max(0.01, x_tag - xf))
    zdotf_x = vf * math.cos(face_tag_heading)
    zdotf_y = vf * math.sin(face_tag_heading)

    M = compute_boundary_matrix(T)
    M_inv = compute_boundary_matrix_inv(T)
    bx = np.array([0.0, float(v0), float(xf), float(zdotf_x)], dtype=np.float64)
    by = np.array([0.0, 0.0,       float(yf), float(zdotf_y)], dtype=np.float64)

    alpha_x = M_inv @ bx
    alpha_y = M_inv @ by
    J = get_mecanum_jacobian()

    # Format analytical polynomial equation strings
    def _poly_str(coeffs, var="t"):
        terms = []
        terms.append(f"{coeffs[0]:+.4f}")
        terms.append(f"{coeffs[1]:+.4f}*{var}")
        terms.append(f"{coeffs[2]:+.4f}*{var}^2")
        terms.append(f"{coeffs[3]:+.4f}*{var}^3")
        return " ".join(terms)

    def _vel_str(coeffs, var="t"):
        return f"{coeffs[1]:+.4f} + ({2.0*coeffs[2]:+.4f})*{var} + ({3.0*coeffs[3]:+.4f})*{var}^2"

    def _acc_str(coeffs, var="t"):
        return f"{2.0*coeffs[2]:+.4f} + ({6.0*coeffs[3]:+.4f})*{var}"

    eq_x = _poly_str(alpha_x)
    eq_xdot = _vel_str(alpha_x)
    eq_xddot = _acc_str(alpha_x)

    eq_y = _poly_str(alpha_y)
    eq_ydot = _vel_str(alpha_y)
    eq_yddot = _acc_str(alpha_y)

    return {
        "d0_m": round(float(d0), 4),
        "bearing_deg": round(math.degrees(bearing_rad), 2),
        "skew_deg": round(math.degrees(skew_rad), 2),
        "standoff_m": round(float(standoff_m), 4),
        "T_s": round(float(T), 4),
        "goal_pose": {"xf_m": round(float(xf), 4), "yf_m": round(float(yf), 4), "heading_deg": round(math.degrees(goal_heading), 2)},
        "boundary_matrix_M": M.tolist(),
        "boundary_matrix_M_inv": M_inv.tolist(),
        "boundary_vector_bx": bx.tolist(),
        "boundary_vector_by": by.tolist(),
        "coefficients_alpha_x": alpha_x.tolist(),
        "coefficients_alpha_y": alpha_y.tolist(),
        "jacobian_J": J.tolist(),
        "motor_params": {"Kv": KV_BACK_EMF, "Ka": KA_INERTIA, "r": WHEEL_RADIUS_R, "Lc": CHASSIS_RADIUS_LC},
        "equations": {
            "x_t": f"x(t) = {eq_x}",
            "xdot_t": f"x_dot(t) = {eq_xdot}",
            "xddot_t": f"x_ddot(t) = {eq_xddot}",
            "y_t": f"y(t) = {eq_y}",
            "ydot_t": f"y_dot(t) = {eq_ydot}",
            "yddot_t": f"y_ddot(t) = {eq_yddot}",
            "speed_v_t": "v(t) = sqrt(x_dot(t)^2 + y_dot(t)^2)",
            "omega_wz_t": "omega(t) = (x_dot*y_ddot - y_dot*x_ddot) / (x_dot^2 + y_dot^2)",
            "motor_voltage_V_i": "V_i(t) = 0.380 * omega_i(t) + 0.060 * (d_omega_i / dt)",
        }
    }


def format_trajectory_report(traj_meta: dict) -> str:
    """
    Renders a clean, formatted ASCII mathematical report of the matrices and equations.
    """
    T = traj_meta["T_s"]
    M = np.array(traj_meta["boundary_matrix_M"])
    M_inv = np.array(traj_meta["boundary_matrix_M_inv"])
    bx = np.array(traj_meta["boundary_vector_bx"])
    by = np.array(traj_meta["boundary_vector_by"])
    ax = np.array(traj_meta["coefficients_alpha_x"])
    ay = np.array(traj_meta["coefficients_alpha_y"])
    J = np.array(traj_meta["jacobian_J"])
    eqs = traj_meta["equations"]
    gp = traj_meta["goal_pose"]

    lines = []
    lines.append("=" * 78)
    lines.append("   KFUPM SCE 672: ADVANCED ROBOTICS — ASSESSMENT 2 (TRAJECTORY MATRICES)")
    lines.append("=" * 78)
    lines.append(f"Initial Tag Distance d0   : {traj_meta['d0_m']:.3f} m")
    lines.append(f"Tag Bearing Angle         : {traj_meta['bearing_deg']:+.2f} deg")
    lines.append(f"Target Surface Skew       : {traj_meta['skew_deg']:+.2f} deg")
    lines.append(f"Trajectory Duration T     : {T:.3f} s")
    lines.append(f"Target Standoff Point     : x_f = {gp['xf_m']:.3f} m, y_f = {gp['yf_m']:.3f} m")
    lines.append("-" * 78)
    lines.append("1. BOUNDARY VALUE SYSTEM: M(T) * alpha = b")
    lines.append("   where z_j(t) = alpha_0 + alpha_1*t + alpha_2*t^2 + alpha_3*t^3")
    lines.append("")
    lines.append("   Boundary Matrix M(T) [4x4]:")
    for row in M:
        lines.append(f"     [ {row[0]:8.4f}  {row[1]:8.4f}  {row[2]:10.4f}  {row[3]:12.4f} ]")
    lines.append("")
    lines.append("   Inverted Boundary Matrix M(T)^(-1) [4x4]:")
    for row in M_inv:
        lines.append(f"     [ {row[0]:8.4f}  {row[1]:8.4f}  {row[2]:10.4f}  {row[3]:12.4f} ]")
    lines.append("")
    lines.append("   Boundary Vectors:")
    lines.append(f"     b_x = [ {bx[0]:.3f},  {bx[1]:.3f},  {bx[2]:.3f},  {bx[3]:.3f} ]^T   (x(0), x_dot(0), x(T), x_dot(T))")
    lines.append(f"     b_y = [ {by[0]:.3f},  {by[1]:.3f},  {by[2]:.3f},  {by[3]:.3f} ]^T   (y(0), y_dot(0), y(T), y_dot(T))")
    lines.append("")
    lines.append("2. SOLVED POLYNOMIAL COEFFICIENT VECTORS:")
    lines.append(f"     alpha_x = M^(-1) * b_x = [ {ax[0]:+.5f},  {ax[1]:+.5f},  {ax[2]:+.5f},  {ax[3]:+.5f} ]^T")
    lines.append(f"     alpha_y = M^(-1) * b_y = [ {ay[0]:+.5f},  {ay[1]:+.5f},  {ay[2]:+.5f},  {ay[3]:+.5f} ]^T")
    lines.append("-" * 78)
    lines.append("3. CLOSED-FORM ANALYTICAL TRAJECTORY EQUATIONS:")
    lines.append(f"   {eqs['x_t']}")
    lines.append(f"   {eqs['xdot_t']}")
    lines.append(f"   {eqs['xddot_t']}")
    lines.append(f"   {eqs['y_t']}")
    lines.append(f"   {eqs['ydot_t']}")
    lines.append(f"   {eqs['yddot_t']}")
    lines.append("")
    lines.append("4. DIFFERENTIAL FLATNESS INVERSION TO ROBOT BODY COMMANDS (Eq. 1):")
    lines.append(f"   Forward Speed    : {eqs['speed_v_t']}")
    lines.append(f"   Yaw Turn Rate    : {eqs['omega_wz_t']}")
    lines.append("   Lateral Velocity : vy(t) = 0.000 m/s (STRICTLY NONHOLONOMIC - NO STRAFING)")
    lines.append("-" * 78)
    lines.append("5. MECANUM INVERSE KINEMATICS JACOBIAN (Eq. 4):")
    lines.append("   [w1, w2, w3, w4]^T = J * [vx, wz]^T,   where r = 0.05m, Lc = 0.20m")
    lines.append("   Jacobian Matrix J [4x2]:")
    for row in J:
        lines.append(f"     [ {row[0]:6.2f}  {row[1]:+6.2f} ]")
    lines.append("-" * 78)
    lines.append("6. MOTOR ARMATURE VOLTAGE MODEL (Eq. 5):")
    lines.append(f"   {eqs['motor_voltage_V_i']}")
    lines.append("   Parameters: Kv = 0.380 V*s/rad, Ka = 0.060 V*s^2/rad")
    lines.append("=" * 78)
    return "\n".join(lines)


def format_initial_poses_report(poses: dict) -> str:
    """
    Renders a formatted ASCII report of the recorded robot pose and tag poses
    captured immediately prior to cubic path planning and execution.
    """
    if not poses or "robot_pose" not in poses:
        return ""
    rp = poses.get("robot_pose", {})
    tt = poses.get("target_tag", {})
    tags = poses.get("all_tags_in_memory", {})
    bc = poses.get("boundary_conditions", {})

    lines = []
    lines.append("=" * 78)
    lines.append("   RECORDED INITIAL POSES BEFORE CUBIC PATH PLANNING & EXECUTION")
    lines.append("=" * 78)
    lines.append(f"Recorded Timestamp       : {poses.get('timestamp_iso', 'N/A')}")
    lines.append(f"Mission State at Capture : {poses.get('mission_state_before_chase', 'N/A')}")
    lines.append("")
    lines.append("1. ROBOT INITIAL 2D WORLD POSE:")
    lines.append(f"   Position (X, Y)       : ({rp.get('x_m', 0.0):+.4f} m, {rp.get('y_m', 0.0):+.4f} m)")
    lines.append(f"   Heading (Yaw)         : {rp.get('yaw_deg', 0.0):+.2f} deg ({rp.get('yaw_rad', 0.0):+.4f} rad)")
    lines.append(f"   Initial Velocity v0   : {rp.get('v0_mps', 0.0):.4f} m/s")
    lines.append("")
    lines.append(f"2. TARGET TAG (T{tt.get('tag_id', '?')}) INITIAL STATE & STANDOFF GOAL:")
    lines.append(f"   Relative Distance d0  : {tt.get('rel_dist_m', 0.0):.4f} m")
    lines.append(f"   Relative Bearing beta : {tt.get('rel_bearing_deg', 0.0):+.2f} deg ({tt.get('rel_bearing_rad', 0.0):+.4f} rad)")
    lines.append(f"   Relative Skew psi     : {tt.get('rel_skew_deg', 0.0):+.2f} deg ({tt.get('rel_skew_rad', 0.0):+.4f} rad)")
    lines.append(f"   Tag World Position    : ({tt.get('world_x_m', 0.0):+.4f} m, {tt.get('world_y_m', 0.0):+.4f} m)")
    lines.append(f"   Standoff Goal (20 cm) : ({tt.get('standoff_world_x_m', 0.0):+.4f} m, {tt.get('standoff_world_y_m', 0.0):+.4f} m)")
    if bc:
        lines.append(f"   Boundary Goal Point   : xf_rel = {bc.get('xf_rel_m', 0.0):+.4f} m, yf_rel = {bc.get('yf_rel_m', 0.0):+.4f} m")
    lines.append("")
    lines.append(f"3. ALL DETECTED APRILTAGS IN SPATIAL MEMORY (Count: {len(tags)}):")
    if tags:
        lines.append(f"   {'Tag':<6} {'World X (m)':<14} {'World Y (m)':<14} {'Rel Dist (m)':<14} {'Rel Brg (deg)':<15} {'Role'}")
        lines.append("   " + "-" * 76)
        for tid, tinfo in tags.items():
            role = "TARGET (PURSUIT)" if tinfo.get("is_target", False) else "Discovered Landmark"
            lines.append(
                f"   T{tinfo.get('tag_id', tid):<5} "
                f"{tinfo.get('world_x_m', 0.0):>+10.4f} m  "
                f"{tinfo.get('world_y_m', 0.0):>+10.4f} m  "
                f"{tinfo.get('rel_dist_m', 0.0):>10.4f} m  "
                f"{tinfo.get('rel_bearing_deg', 0.0):>+9.2f} deg  "
                f"{role}"
            )
    else:
        lines.append("   (No tags in memory)")
    lines.append("=" * 78)
    return "\n".join(lines)


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
    vf: float = 0.05,
):
    """
    Plans a 2D flat cubic trajectory z(t) = [x(t), y(t)]^T in the robot's initial chase frame
    that moves from (0, 0) to the 20 cm standoff point directly in front of the tag,
    curving smoothly so the robot arrives directly aligned and perpendicular to the tag.
    Terminal velocity tangent vector is constrained along the tag normal:
        xdot(T) = vf * cos(goal_heading)
        ydot(T) = vf * sin(goal_heading)
    """
    T = compute_time_horizon(d0, v_max=v_max)

    # Tag position in robot frame at start of chase
    x_tag = d0 * math.cos(bearing_rad)
    y_tag = d0 * math.sin(bearing_rad)

    # Desired final heading: point directly at the tag center from standoff
    goal_heading = wrap_angle(bearing_rad - 0.25 * skew_rad) if abs(skew_rad) > 0.04 else bearing_rad

    # Standoff point 20 cm in front of the tag along goal_heading
    xf = max(0.08, x_tag - standoff_m * math.cos(goal_heading))
    yf = y_tag - standoff_m * math.sin(goal_heading)

    # Terminal tangent velocity components aligned directly to face the tag
    face_tag_heading = math.atan2(y_tag - yf, max(0.01, x_tag - xf))
    zdotf_x = vf * math.cos(face_tag_heading)
    zdotf_y = vf * math.sin(face_tag_heading)

    # Boundary conditions:
    # Start (t=0): x(0)=0, y(0)=0, moving straight forward: xdot(0)=v0, ydot(0)=0
    # End   (t=T): x(T)=xf, y(T)=yf, arriving aligned with goal_heading tangent: xdot(T)=zdotf_x, ydot(T)=zdotf_y
    alpha_x = solve_flat_polynomial(0.0, v0, xf, zdotf_x, T)
    alpha_y = solve_flat_polynomial(0.0, 0.0, yf, zdotf_y, T)
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
