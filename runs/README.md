# Assessment 2: Experimental Runs Catalog & Trajectory Matrix Database

This directory maintains complete telemetry archives, generated figures, solved polynomial coefficient vectors, and boundary matrix inversions for every mission run.

## Summary of Recorded Runs

| Run ID | Target | Initial Dist | Final Standoff | Duration | Peak Speed | Solved Cubic Equation $x(t)$ | Matrices & Coefficients | 2D Path Map | Velocity & Voltage | Telemetry Data |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **run_01_tag3_hardware** | Tag 3 | 1.50 m | **0.199 m** | 29.6 s | 0.280 m/s | `x(t) = +0.0000 +0.0600*t +0.1082*t^2 -0.0139*t^3` | [View Matrices & Equations](./run_01_tag3_hardware/trajectory_and_matrices.txt) | [View 2D Map](./run_01_tag3_hardware/path_2d_planned_vs_actual.png) | [View Velocity Plot](./run_01_tag3_hardware/velocity_and_voltage_profile.png) | [CSV Data](./run_01_tag3_hardware/) |
| **run_20261028_112335_tag3** | Tag 3 | 1.50 m | **0.206 m** | 28.0 s | 0.280 m/s | `x(t) = +0.0000 +0.0800*t -0.0260*t^2 +0.0027*t^3` | [View Matrices & Equations](./run_20261028_112335_tag3/trajectory_and_matrices.txt) | [View 2D Map](./run_20261028_112335_tag3/path_2d_planned_vs_actual.png) | [View Velocity Plot](./run_20261028_112335_tag3/velocity_and_voltage_profile.png) | [CSV Data](./run_20261028_112335_tag3/) |

---
## Mathematical Trajectory & Matrix Formulation

For each run, the differential flatness boundary value problem is solved at runtime:
$$M(T) \boldsymbol{\alpha}_j = \mathbf{b}_j, \quad j \in \{x, y\}$$

where the $4 \times 4$ boundary condition matrix and its inverse are:
$$M(T) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 1 & T & T^2 & T^3 \\ 0 & 1 & 2T & 3T^2 \end{bmatrix}, \quad M(T)^{-1} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ -\frac{3}{T^2} & -\frac{2}{T} & \frac{3}{T^2} & -\frac{1}{T} \\ \frac{2}{T^3} & \frac{1}{T^2} & -\frac{2}{T^3} & \frac{1}{T^2} \end{bmatrix}$$

The Mecanum inverse kinematics mapping strictly enforces nonholonomic motion (lateral speed $v_y = 0.0$):
$$\begin{bmatrix} \omega_1 \\ \omega_2 \\ \omega_3 \\ \omega_4 \end{bmatrix} = \frac{1}{r} \begin{bmatrix} 1 & L_c \\ 1 & -L_c \\ 1 & -L_c \\ 1 & L_c \end{bmatrix} \begin{bmatrix} v_x \\ \omega_z \end{bmatrix} = \begin{bmatrix} 20.0 & 4.0 \\ 20.0 & -4.0 \\ 20.0 & -4.0 \\ 20.0 & 4.0 \end{bmatrix} \begin{bmatrix} v_x \\ \omega_z \end{bmatrix}$$

The motor armature terminal voltages are evaluated dynamically:
$$V_i(t) = K_v \omega_i(t) + K_a \dot{\omega}_i(t), \quad K_v = 0.380\text{ V}\cdot\text{s/rad}, \; K_a = 0.060\text{ V}\cdot\text{s}^2\text{/rad}$$

---
## Automated Logging for Future Runs
Whenever a new run is executed via `mission_node.py` or `mission_runner_win.py`, a new timestamped folder (`runs/run_YYYYMMDD_HHMMSS_tag<ID>/`) is automatically populated with:
1. `trajectory_and_matrices.txt` (Complete ASCII report of boundary matrices, inverted matrices, coefficient vectors, and closed-form equations)
2. `run_summary.json` (Structured JSON containing all matrix arrays and mission performance metrics)
3. `path_2d_planned_vs_actual.png` (2D trajectory map with annotated cubic equation)
4. `velocity_and_voltage_profile.png` (Velocity and motor voltage profiles with mathematical annotations)
5. `motor_voltage_log.csv` (1-second Table 1 telemetry log)
6. `motor_voltage_continuous.csv` (20 Hz high-resolution state stream)
