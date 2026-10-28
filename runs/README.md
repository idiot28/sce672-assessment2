# Assessment 2: Experimental Runs Catalog

This directory maintains complete telemetry archives and generated figures for every mission run.

## Summary of Recorded Runs

| Run ID | Target Tag | Initial Dist | Final Standoff | Duration | Peak Speed | 2D Path Map | Velocity & Voltage | Telemetry Data |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **run_01_tag3_hardware** | Tag 3 | 1.50 m | **0.199 m** | 29.6 s | 0.280 m/s | [View 2D Map](./run_01_tag3_hardware/path_2d_planned_vs_actual.png) | [View Velocity Plot](./run_01_tag3_hardware/velocity_and_voltage_profile.png) | [CSV Data](./run_01_tag3_hardware/) |

---
## Automated Logging for Future Runs
Whenever a new run is executed via `mission_node.py` or `mission_runner_win.py`, a new timestamped folder (`runs/run_YYYYMMDD_HHMMSS_tag<ID>/`) is automatically populated with:
1. `motor_voltage_log.csv` (1-second Table 1 telemetry)
2. `motor_voltage_continuous.csv` (20 Hz high-resolution state stream)
3. `path_2d_planned_vs_actual.png` (2D trajectory map)
4. `velocity_and_voltage_profile.png` (Velocity and motor voltage profiles)
5. `run_summary.json` (Run metrics)
