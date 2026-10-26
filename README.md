# SCE 672: Advanced Robotics - Assessment 2
## Differential Flatness Pursuit and Motor Voltage Dynamics

**Author:** Raian E Rabbani (Student ID: 202617880)  
**Institution:** King Fahd University of Petroleum & Minerals (KFUPM)  
**Instructor:** Prof. Aamer Iqbal Bhatti  
**Target:** AprilTag $T_3$ ($d_0 = 1.50$~m, Standoff = 20~cm)

---

### Project Overview
Autonomous AprilTag tracking and precision 20 cm standoff braking using differential flatness inversion and nonholonomic kinematics ($v_y = 0$) on a DJI RoboMaster EP robot.

### Repository Contents
- **`rm_flatness_nav/`**: Complete ROS 2 Humble package containing:
  - `mission_node.py`: Supervisory FSM (`SEARCHING` $\rightarrow$ `ALIGNING` $\rightarrow$ `CHASING` $\rightarrow$ `ARRIVED`), cubic flatness planner, optical distance tracking, and motor voltage model ($K_v = 0.38, K_a = 0.06$).
  - `vision_tof.py`: Calibrated pinhole optics and AprilTag detector.
  - `flatness.py`: Closed-form cubic polynomial solver and flatness inversion equations.
  - `launch/flatness_pursuit.launch.py`: ROS 2 launch file.
- **`motor_voltage_log.csv`**: Exported 1-second Table 1 experimental telemetry log.
- **`motor_voltage_continuous.csv`**: 20 Hz high-resolution trajectory and motor voltage log.
- **`debug_gui.py`**: Real-time Windows Tkinter dashboard with live parameter sliders and WASD teleop.
- **`report.pdf` / `Raian_SCE672_Assessment2.pdf`**: Complete 4-page LaTeX report with theoretical derivations, Table 1, and execution plots.
- **`run_v4.bat` & `run_wsl.sh`**: One-click build and execution scripts.

---

### How to Build & Run

#### 1. Quick Launch (Windows)
Double-click `run_v4.bat` and select **Option 1**:
```cmd
.\run_v4.bat
```

#### 2. Manual Launch (WSL2 / ROS 2 Humble)
```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Build the package
colcon build --packages-select rm_flatness_nav --symlink-install
source install/setup.bash

# Run mission targeting Tag 3
ros2 launch rm_flatness_nav flatness_pursuit.launch.py conn_type:=ap target_tag_id:=3
```

#### 3. Real-Time Debug GUI (Windows)
```powershell
python debug_gui.py
```
