# SCE 672: Advanced Robotics - Assessment 2
## Differential Flatness Pursuit and Motor Voltage Dynamics

**Author:** Raian E Rabbani (Student ID: 202617880)  
**Institution:** King Fahd University of Petroleum & Minerals (KFUPM)  
**Instructor:** Prof. Aamer Iqbal Bhatti  
**Target:** AprilTag $T_3$ ($d_0 = 1.50$~m, Standoff = 20~cm)

---

### Project Overview
Autonomous AprilTag tracking and precision 20 cm standoff braking using differential flatness inversion and nonholonomic kinematics ($v_y = 0$) on a DJI RoboMaster EP robot.

### Repository Structure & Compartments

```
project apriltag v4/
│
├── run_v4.bat                    # [ENTRY] Unified Windows & WSL Interactive Launcher
├── run_win.bat                   # [ENTRY] Direct Native Windows Autonomous Runner
├── run_wsl.sh                    # [ENTRY] Direct WSL2 / ROS 2 Humble Build & Launch Script
├── Raian_SCE672_Assessment2.pdf  # Formal Submission PDF Report (4 Pages, IEEE Format)
├── README.md                     # Project Overview, Setup & User Guide
├── MASTER_STATUS.md              # Technical Architecture, Verification & Status
│
├── src/                          # Native Python Modules & GUI
│   ├── debug_gui.py              # Real-Time 40 FPS Control Dashboard & WASD Teleop
│   ├── mission_runner_win.py     # Main Windows Autonomous Mission Node
│   ├── flatness_math.py          # Differential Flatness & Mecanum Kinematics Math
│   ├── vision_camera.py          # Optimized AprilTag 36h11 Optical Detector (5.5 ms)
│   ├── run_archiver.py           # Automated Run Archiving & Plot Generation Engine
│   └── plot_clean_figure.py      # Publication Figure Generator
│
├── ros2/                         # ROS 2 Humble Workspace Package
│   └── rm_flatness_nav/          # Full ROS 2 Python Package (setup.py, launch, node)
│       ├── launch/flatness_pursuit.launch.py
│       └── rm_flatness_nav/
│           ├── mission_node.py
│           ├── vision_tof.py
│           └── flatness_math.py
│
├── report/                       # Academic Deliverables & Publication Assets
│   ├── report.tex                # 4-Page IEEE LaTeX Assessment 2 Report
│   ├── report.pdf                # Compiled 4-Page PDF Report
│   ├── Raian_SCE672_Assessment2.pdf # Formal Submission PDF
│   ├── fig_path_2d_planned_vs_actual.png
│   ├── fig_speed_and_voltage.png
│   └── fig_all_motors_and_flatness.png
│
├── runs/                         # Persistent Run Archives (Plots, Logs, JSON metadata)
│   ├── README.md                 # Master Run Catalog
│   └── run_01_tag3_hardware/     # Hardware Experiment 1
│       ├── path_2d_planned_vs_actual.png
│       ├── velocity_and_voltage_profile.png
│       ├── motor_voltage_log.csv
│       ├── motor_voltage_continuous.csv
│       └── run_summary.json
│
├── config/                       # Runtime Configuration & Tuning Parameters
│   └── tuning_params.json        # Default Autonomous Pursuit & Geometry Tunings
│
└── vendor/                       # Third-party wheels & dependencies
    └── robomaster.whl            # Offline RoboMaster Python SDK wheel
```

---

### How to Build & Run

#### 1. Quick Launch (Windows One-Click)
Double-click `run_v4.bat` and select **Option 1**:
```cmd
.\run_v4.bat
```

#### 2. Manual Launch (WSL2 / ROS 2 Humble)
```bash
bash run_wsl.sh ap 3 false 0.38
```

#### 3. Real-Time Debug GUI (Windows)
```powershell
python src/debug_gui.py
```
