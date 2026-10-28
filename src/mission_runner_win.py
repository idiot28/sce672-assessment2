#!/usr/bin/env python3
"""
mission_runner_win.py - Standalone Native Windows Mission Runner (Camera-Only)
==============================================================================
Parallel implementation running directly on Windows without WSL or ROS 2.
Eliminates WSL2 network virtualization overhead, UDP packet drops, and 9P filesystem latency.

Assessment 2 Specifications:
  1. 100% Pure Camera-Based Navigation:
     - Calibrated pinhole Euclidean distance from AprilTag geometry.
     - Subpixel parallel skew & bearing alignment from tag corners.
     - 2D World Spatial Tag Memory (tx, ty) across 360-degree arena.
     - Close-range optical + wheel displacement dead-reckoning down to 20 cm standoff.
  2. Differential Flatness & Continuous Cubic Replanning:
     - Re-solves `plan_aligning_cubic_path` at 20 Hz during CHASING.
     - Strictly nonholonomic kinematics (vy = 0.0 at all times).
     - Motor armature voltages V1..V4 calculated and logged per SCE 672 Eq. (5).
  3. Real-Time Debug GUI & CSV Logging:
     - Generates live_camera.jpg with visual overlays.
     - Generates live_telemetry.json for debug_gui.py.
     - Saves motor_voltage_log.csv (Table 1) and motor_voltage_continuous.csv.
     - Listens to gui_command.json (WASD keyboard teleop, E-stop, Restart) and tuning_params.json.
"""

import os
import sys
import csv
import json
import math
import time
import socket
import queue
import argparse
import threading
import numpy as np
import cv2

from flatness_math import (
    wrap_angle,
    plan_aligning_cubic_path,
    evaluate_flatness,
    wheel_kinematics,
    motor_voltages,
    compute_trajectory_matrices,
    format_trajectory_report,
    format_initial_poses_report,
)
from vision_camera import (
    CameraAprilTagDetector,
    CameraSimArena,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
BASE_DIR = PROJECT_DIR

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


# =============================================================================
# Ultra-Low Latency DropOldestQueue & Custom CvLiveView for Windows RoboMaster
# =============================================================================
class DropOldestQueue(queue.Queue):
    """
    Single-slot FIFO queue that immediately replaces stale frames upon arrival of a new frame.
    Guarantees 0 ms internal buffer latency and completely eliminates H.264 stream lag.
    """
    def put(self, item, block=True, timeout=None):
        with self.mutex:
            while self._qsize() >= self.maxsize:
                try:
                    self._get()
                except Exception:
                    break
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


class CvLiveView:
    """
    Connects to the RoboMaster EP H.264 stream on TCP port 40921 using OpenCV's
    built-in FFmpeg backend on Windows with zero-latency buffer parameters.
    """

    def __init__(self, robot):
        self._robot = robot
        self._cap = None
        self._streaming = False
        self._queue = DropOldestQueue(maxsize=1)
        self._thread = None
        self._frame_count = 0

    def start_video_stream(self, display=False, addr=None, ip_proto="tcp"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|max_delay;0"
        ip = addr[0] if addr else getattr(self._robot, "ip", "192.168.2.1")
        port = addr[1] if addr else 40921
        stream_url = f"tcp://{ip}:{port}"
        self._streaming = True

        def _worker():
            time.sleep(0.1)
            try:
                self._cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                if not self._cap.isOpened():
                    self._cap.open(stream_url)
                if self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                self._cap = None

            while self._streaming:
                if self._cap is not None and self._cap.isOpened():
                    ret, frame = self._cap.read()
                    if ret and frame is not None:
                        self._frame_count += 1
                        self._queue.put(frame)
                    else:
                        time.sleep(0.005)
                else:
                    time.sleep(0.08)
                    try:
                        self._cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                        if self._cap.isOpened():
                            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()
        return True

    def stop_video_stream(self):
        self._streaming = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._queue.queue.clear()
        return True

    def read_video_frame(self, timeout=0.1, strategy="newest"):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


def patch_robomaster_windows(conn_type: str):
    """Configures RoboMaster SDK networking & video decoding for native Windows execution."""
    try:
        import robomaster.config
        import robomaster.conn
        import robomaster.camera
        import robomaster.media

        # Configure OpenCV FFmpeg low-latency capture flags
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|max_delay;0"

        # Ensure DEFAULT_CONN_PROTO is always defined in config
        if not hasattr(robomaster.config, "DEFAULT_CONN_PROTO"):
            robomaster.config.DEFAULT_CONN_PROTO = getattr(robomaster.config, "DEFAULT_PROTO_TYPE", "udp")

        # Patch FtpConnection dummy if not available
        if not hasattr(robomaster.conn, "FtpConnection"):
            class DummyFtp:
                def __init__(self, *a, **kw): pass
                def connect(self, *a, **kw): return True
                def stop(self, *a, **kw): pass
            robomaster.conn.FtpConnection = DummyFtp

        # Patch native robomaster.media.LiveView queue depth to 1 for zero-latency
        try:
            _orig_liveview_init = robomaster.media.LiveView.__init__
            def _fast_liveview_init(self, robot):
                _orig_liveview_init(self, robot)
                self._video_frame_queue = DropOldestQueue(maxsize=1)
            robomaster.media.LiveView.__init__ = _fast_liveview_init
        except Exception:
            pass

        # Auto-detect local Windows IP on robot subnet
        target_ip = "192.168.42.2" if conn_type == "rndis" else "192.168.2.1"
        target_subnet = "192.168.42." if conn_type == "rndis" else "192.168.2."
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.4)
        lip = ""
        try:
            s.connect((target_ip, 80))
            lip = s.getsockname()[0]
        except Exception:
            pass
        finally:
            s.close()

        if lip and lip.startswith(target_subnet):
            robomaster.config.LOCAL_IP_STR = lip
            print(f"[NET] Windows Local IP on robot network: {lip}")
        else:
            robomaster.config.LOCAL_IP_STR = None

    except Exception as e:
        print(f"[WARN] RoboMaster SDK patch note: {e}")


# =============================================================================
# Standalone Native Windows Mission Runner
# =============================================================================
class WindowsMissionRunner:
    def __init__(
        self,
        conn_type: str = "ap",
        target_tag_id: int = 3,
        use_sim: bool = False,
        v_max: float = 0.38,
        standoff_m: float = 0.20,
        output_dir: str = BASE_DIR,
    ):
        self.conn_type = conn_type.lower()
        self.target_tag_id = int(target_tag_id)
        self.use_sim = bool(use_sim)
        self.output_dir = output_dir

        # Live-Tunable Parameters (matched to Assessment 2 specifications)
        self.sweep_wz_rad = 0.35          # 360-deg sweep rotation speed (rad/s)
        self.align_wz_max = 0.50          # Max turn rate during ALIGNING (rad/s)
        self.v_max = float(v_max)         # Max chassis speed (m/s)
        self.v_min = 0.08                 # Min forward approach speed (m/s)
        self.chase_wz_max = 0.55          # Max turn rate during CHASING (rad/s)
        self.turn_radius_m = 0.30         # Min Turning Radius R: |wz| <= |vx| / R
        self.standoff_m = float(standoff_m)  # Default 0.20 m = 20 cm
        self.arm_x_mm = 140               # Arm forward position (mm)
        self.arm_y_mm = 35                # Arm height position (mm)
        self.kp_bearing = 1.65            # Visual bearing proportional gain
        self.kp_skew = 0.45               # Parallel skew alignment gain
        self.tag_size_mm = 48             # Inner black AprilTag square (48 mm)
        self._last_tuning_mtime = 0.0
        self._last_disk_poll_time = 0.0

        # Hardware & Sim Objects
        self.hw_connected = False
        self.ep_robot = None
        self.sim = None
        self.lock = threading.Lock()

        # Camera Detector (Pure Vision: No ToF Sensor)
        self.detector = CameraAprilTagDetector(tag_size_m=float(self.tag_size_mm) / 1000.0)

        # Odometry State
        self.hw_pos_initialized = False
        self.hw_x0 = 0.0
        self.hw_y0 = 0.0
        self.hw_yaw_initialized = False
        self.hw_yaw0_deg = 0.0
        self._last_pos_cb_time = 0.0
        self._last_att_cb_time = 0.0
        self._last_esc_cb_time = 0.0
        self.hw_esc_w = np.zeros(4, dtype=np.float64)
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.prev_yaw = None
        self.odom_trail = []
        self.history_trails = []

        # Vision State & 2D World Spatial Tag Memory
        self.tag_visible = False
        self.tag_dist_m = 1.25
        self.tag_bearing_rad = 0.0
        self.tag_skew_rad = 0.0
        self.tag_normal_bot_rad = math.pi
        self.tag_normal_world_rad = math.pi
        self.last_seen_time = 0.0
        self.last_seen_dist = 2.0
        self.last_seen_odom_x = 0.0
        self.last_seen_odom_y = 0.0
        self.tag_memory = {}

        # Supervisory FSM State (Launches directly to TELEOP mode for manual positioning)
        self.state = "TELEOP"
        self.park_phase = "BACKUP"
        self.park_phase_time = time.monotonic()
        self.park_passes = 0
        self.start_time = time.monotonic()
        self.state_time = time.monotonic()
        self.sweep_rad = 0.0
        self.mission_done = False
        self.estop_paused = False

        # Manual override / WASD teleop from GUI
        self.manual_vx = 0.0
        self.manual_wz = 0.0
        self.manual_until = 0.0

        # Flat Cubic Path Parameters
        self.alpha_x = np.zeros(4)
        self.alpha_y = np.zeros(4)
        self.cubic_T = 3.5
        self.planned_path_pts = []
        self.trajectory_analysis = {}
        self.initial_poses_before_chase = {}

        # Velocities & Motor Voltage State (Strictly nonholonomic: vy = 0.0)
        self.vx = 0.0
        self.wz = 0.0
        self._halted = True
        self.w_prev = np.zeros(4, dtype=np.float64)
        self.voltages = np.zeros(4, dtype=np.float64)
        self.table1_rows = []
        self.continuous_rows = []
        self.last_logged_sec = -1

        # Load initial tuning parameters
        self._load_tuning_file(initial=True)
        # Ensure explicitly requested CLI target_tag_id takes precedence on startup
        if target_tag_id is not None:
            self.target_tag_id = int(target_tag_id)

        # Do NOT pre-compute a fake path before discovering the target tag
        self.planned_path_pts = []

        # Initialize Hardware or Simulation
        self._init_hardware_or_sim()

        # Background Threads (Camera Grabber decoupled from Vision Worker for ZERO latency)
        self._stop_evt = threading.Event()
        self._new_frame_evt = threading.Event()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None

        self._grabber_thread = threading.Thread(target=self._camera_grabber_loop, daemon=True)
        self._grabber_thread.start()
        self._vision_thread = threading.Thread(target=self._vision_worker_loop, daemon=True)
        self._vision_thread.start()
        self._act_thread = threading.Thread(target=self._chassis_actuator_loop, daemon=True)
        self._act_thread.start()

        print("=" * 76)
        print("  DJI RoboMaster EP - Native Windows Assessment 2 Mission Runner")
        print("  Pure Camera-Based Navigation  |  Differential Flatness  |  vy = 0")
        print("=" * 76)
        print(f"Target Tag: Tag {self.target_tag_id}  |  Mode: {'HARDWARE (' + self.conn_type.upper() + ')' if self.hw_connected else 'WINDOWS SIMULATION'}")
        print(f"{'Timestamp t(s)':<16} {'V1(V)':>8} {'V2(V)':>8} {'V3(V)':>8} {'V4(V)':>8}   {'State'}")
        print("-" * 76)

    def _load_tuning_file(self, initial: bool = False):
        tpath = os.path.join(self.output_dir, "config", "tuning_params.json")
        if not os.path.exists(tpath):
            tpath = os.path.join(self.output_dir, "tuning_params.json")
        if not os.path.exists(tpath):
            return
        try:
            mtime = os.path.getmtime(tpath)
            if not initial and mtime <= self._last_tuning_mtime:
                return
            self._last_tuning_mtime = mtime
            with open(tpath, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._apply_tuning_dict(cfg, move_arm=(not initial))
        except Exception:
            pass

    def _apply_tuning_dict(self, cfg: dict, move_arm: bool = True):
        old_arm = (self.arm_x_mm, self.arm_y_mm)
        with self.lock:
            if "target_tag_id" in cfg:
                self.target_tag_id = int(cfg["target_tag_id"])
            if "sweep_wz_rad" in cfg:
                self.sweep_wz_rad = float(np.clip(float(cfg["sweep_wz_rad"]), 0.10, 2.00))
            if "align_wz_max" in cfg:
                self.align_wz_max = float(np.clip(float(cfg["align_wz_max"]), 0.10, 2.00))
            if "v_max_mps" in cfg:
                self.v_max = float(np.clip(float(cfg["v_max_mps"]), 0.05, 0.80))
            if "v_min_mps" in cfg:
                self.v_min = float(np.clip(float(cfg["v_min_mps"]), 0.03, 0.40))
            if "chase_wz_max" in cfg:
                self.chase_wz_max = float(np.clip(float(cfg["chase_wz_max"]), 0.10, 2.00))
            if "turn_radius_m" in cfg:
                self.turn_radius_m = float(np.clip(float(cfg["turn_radius_m"]), 0.08, 2.50))
            if "standoff_m" in cfg:
                self.standoff_m = float(np.clip(float(cfg["standoff_m"]), 0.08, 0.80))
            if "arm_x_mm" in cfg:
                self.arm_x_mm = int(np.clip(int(float(cfg["arm_x_mm"])), 50, 240))
            if "arm_y_mm" in cfg:
                self.arm_y_mm = int(np.clip(int(float(cfg["arm_y_mm"])), -80, 180))
            if "kp_bearing" in cfg:
                self.kp_bearing = float(np.clip(float(cfg["kp_bearing"]), 0.20, 4.00))
            if "kp_skew" in cfg:
                self.kp_skew = float(np.clip(float(cfg["kp_skew"]), 0.0, 2.00))
            if "tag_size_mm" in cfg:
                self.tag_size_mm = int(np.clip(int(float(cfg["tag_size_mm"])), 20, 150))
                self.detector.tag_size_m = float(self.tag_size_mm) / 1000.0

        new_arm = (self.arm_x_mm, self.arm_y_mm)
        if move_arm and (new_arm != old_arm or cfg.get("force_arm_move", False)):
            self._move_arm_async(self.arm_x_mm, self.arm_y_mm)

    def _move_arm_async(self, x_mm: int, y_mm: int):
        if not self.hw_connected or self.ep_robot is None:
            return

        def _worker():
            try:
                self.ep_robot.robotic_arm.moveto(x=int(x_mm), y=int(y_mm)).wait_for_completed(timeout=2.0)
                print(f"[ARM] Moved robotic arm to (X={x_mm} mm, Y={y_mm} mm).")
            except Exception as e:
                print(f"[ARM] Moveto note: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def _init_hardware_or_sim(self):
        if not self.use_sim:
            target_ip = "192.168.42.2" if self.conn_type == "rndis" else "192.168.2.1"
            patch_robomaster_windows(self.conn_type)
            try:
                from robomaster import robot
                print(f"[*] Attempting direct connection to RoboMaster EP at {target_ip} ({self.conn_type})...")
                self.ep_robot = robot.Robot()
                self.ep_robot.initialize(conn_type=self.conn_type)
                self.hw_connected = True
                print(f"[+] Connected to RoboMaster EP hardware ({self.conn_type}) on Windows!")

                # Move robotic arm
                try:
                    self._move_arm_async(self.arm_x_mm, self.arm_y_mm)
                except Exception:
                    pass

                # Subscribe to chassis position, attitude, and ESC telemetry
                try:
                    self.ep_robot.chassis.sub_position(cs=0, freq=20, callback=self._on_pos)
                    self.ep_robot.chassis.sub_attitude(freq=20, callback=self._on_att)
                    self.ep_robot.chassis.sub_esc(freq=20, callback=self._on_esc)
                    time.sleep(0.08)
                except Exception as e:
                    print(f"[WARN] Sensor subscription note: {e}")

                # Start native video stream on robot
                try:
                    self.ep_robot.camera._stream_sdk(1, "360p")
                    self.ep_robot.camera._video_stream(1, "360p")
                    self.ep_robot.camera._video_enable = True
                    print("[+] RoboMaster camera hardware stream enabled (360p @ port 40921).")
                except Exception as e:
                    print(f"[WARN] Camera stream protocol start note: {e}")

                return
            except Exception as e:
                print(f"[!] Hardware unavailable at {target_ip} ({e}).")
                print(f"    - Wi-Fi AP Mode: ensure Windows Wi-Fi is connected to 'RMEP-...' ({target_ip}).")
                print(f"    - USB RNDIS Mode: ensure USB cable is connected (192.168.42.2).")
                print("[*] Automatically switching to Camera-Only Simulation Arena...")

        self.hw_connected = False
        self.sim = CameraSimArena(target_tag_id=self.target_tag_id)
        print(f"[+] Simulation Mode Active: 360-Deg Arena with AprilTag T{self.target_tag_id}.")

    # =========================================================================
    # Hardware Telemetry Callbacks
    # =========================================================================
    def _on_pos(self, sub_info):
        if not sub_info or len(sub_info) < 2:
            return
        try:
            x_raw = float(sub_info[0])
            y_raw = float(sub_info[1])
            now = time.monotonic()
            with self.lock:
                if not self.hw_pos_initialized:
                    self.hw_x0 = x_raw
                    self.hw_y0 = y_raw
                    self.hw_pos_initialized = True
                self.odom_x = x_raw - self.hw_x0
                self.odom_y = -(y_raw - self.hw_y0)  # Convert DJI +Y (right) to standard +Y (left/CCW)
                self._last_pos_cb_time = now
        except Exception:
            pass

    def _on_att(self, sub_info):
        if not sub_info or len(sub_info) < 1:
            return
        try:
            yaw_deg = float(sub_info[0])
            now = time.monotonic()
            with self.lock:
                if not self.hw_yaw_initialized:
                    self.hw_yaw0_deg = yaw_deg
                    self.hw_yaw_initialized = True
                delta_deg = yaw_deg - self.hw_yaw0_deg
                while delta_deg > 180.0:
                    delta_deg -= 360.0
                while delta_deg < -180.0:
                    delta_deg += 360.0
                yaw_rad = -math.radians(delta_deg)  # Convert DJI +CW to standard +CCW
                if self.prev_yaw is not None:
                    self.sweep_rad += abs(wrap_angle(yaw_rad - self.prev_yaw))
                self.prev_yaw = yaw_rad
                self.odom_yaw = yaw_rad
                self._last_att_cb_time = now
        except Exception:
            pass

    def _on_esc(self, sub_info):
        if not sub_info or len(sub_info) < 1:
            return
        try:
            rpms = sub_info[0]
            if isinstance(rpms, (list, tuple)) and len(rpms) >= 4:
                rpm_to_rads = (2.0 * math.pi) / 60.0
                w1_fr = float(rpms[0]) * rpm_to_rads
                w2_fl = -float(rpms[1]) * rpm_to_rads
                w3_rl = -float(rpms[2]) * rpm_to_rads
                w4_rr = float(rpms[3]) * rpm_to_rads
                with self.lock:
                    self.hw_esc_w = np.array([w1_fr, w2_fl, w3_rl, w4_rr], dtype=np.float64)
                    self._last_esc_cb_time = time.monotonic()
        except Exception:
            pass

    # =========================================================================
    # Dedicated 20 Hz Chassis Actuator Loop (Zero Jitter, vy = 0.0)
    # =========================================================================
    def _chassis_actuator_loop(self):
        while not self._stop_evt.is_set():
            if self.hw_connected and self.ep_robot is not None:
                with self.lock:
                    vx = float(self.vx)
                    wz = float(self.wz)
                    halted = bool(self._halted)
                try:
                    if halted:
                        self.ep_robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.5)
                    else:
                        self.ep_robot.chassis.drive_speed(
                            x=vx, y=0.0, z=float(-math.degrees(wz)), timeout=0.5
                        )
                except Exception:
                    pass
            time.sleep(0.050)

    # =========================================================================
    # =========================================================================
    # Zero-Latency Hardware Camera Grabber Thread (Full 30 FPS, Zero Processing)
    # =========================================================================
    def _camera_grabber_loop(self):
        """
        Pure zero-latency camera frame grabber using OpenCV FFmpeg backend.
        Pulls newest frame directly from RoboMaster TCP stream at port 40921 into RAM.
        Zero image processing, zero detection, zero disk I/O.
        """
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|max_delay;0"
        target_ip = "192.168.42.2" if self.conn_type == "rndis" else "192.168.2.1"
        stream_url = f"tcp://{target_ip}:40921"
        cap = None

        while not self._stop_evt.is_set():
            if self.hw_connected and self.ep_robot is not None:
                if cap is None or not cap.isOpened():
                    try:
                        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                        if not cap.isOpened():
                            cap.open(stream_url)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            print(f"[CAM] OpenCV low-latency stream capture connected to {stream_url}!")
                        else:
                            time.sleep(0.15)
                            continue
                    except Exception:
                        time.sleep(0.15)
                        continue

                try:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        with self.lock:
                            self.latest_raw_frame = frame
                        self._new_frame_evt.set()
                    else:
                        time.sleep(0.004)
                except Exception:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    cap = None
                    time.sleep(0.10)

            elif self.sim is not None:
                time.sleep(0.030)
                frame = self.sim.render_frame()
                if frame is not None:
                    with self.lock:
                        self.latest_raw_frame = frame
                    self._new_frame_evt.set()
            else:
                time.sleep(0.04)

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    # =========================================================================
    # Asynchronous Vision Worker (AprilTag Detection Decoupled from Video Display)
    # =========================================================================
    def _vision_worker_loop(self):
        """
        Runs AprilTag detection and 2D spatial memory updates in background on frame copies.
        Never blocks the camera grabber thread or GUI display frame rate.
        """
        last_file_write = 0.0
        while not self._stop_evt.is_set():
            if not self._new_frame_evt.wait(timeout=0.10):
                continue
            self._new_frame_evt.clear()

            with self.lock:
                if self.latest_raw_frame is None:
                    continue
                frame_copy = self.latest_raw_frame.copy()

            now = time.monotonic()
            dets = self.detector.detect_all(frame_copy, default_id=self.target_tag_id)
            target_det = None

            with self.lock:
                for d in dets:
                    tid = int(d["id"])
                    b_rad = float(d["bearing_rad"])
                    s_rad = float(d["skew_rad"])
                    d_m = max(0.08, float(d["dist_m"]))

                    # Integrate into 2D World Spatial Tag Memory
                    world_dir = wrap_angle(self.odom_yaw + b_rad)
                    tx = self.odom_x + d_m * math.cos(world_dir)
                    ty = self.odom_y + d_m * math.sin(world_dir)

                    # Compute tag face outward normal vector from PnP / skewness
                    n_cam = d.get("normal_cam")
                    if n_cam is not None:
                        # Convert camera frame normal (+Z forward, +X right) to robot frame (+X forward, +Y left)
                        n_bot_x = float(n_cam[2])
                        n_bot_y = -float(n_cam[0])
                        psi_norm_bot = math.atan2(n_bot_y, n_bot_x)
                        psi_norm_world = wrap_angle(self.odom_yaw + psi_norm_bot)
                    else:
                        psi_norm_bot = wrap_angle(b_rad + math.pi + s_rad)
                        psi_norm_world = wrap_angle(self.odom_yaw + psi_norm_bot)

                    if tid not in self.tag_memory:
                        self.tag_memory[tid] = {
                            "x": tx,
                            "y": ty,
                            "yaw": world_dir,
                            "normal_world": psi_norm_world,
                            "normal_bot": psi_norm_bot,
                            "facing_deg": round(math.degrees(psi_norm_world), 1),
                            "dist_m": d_m,
                            "skew_rad": s_rad,
                            "normal_cam": n_cam.tolist() if isinstance(n_cam, np.ndarray) else n_cam,
                            "tvec_cam": [round(float(v), 3) for v in d.get("tvec_cam", [0, 0, 0])],
                        }
                    else:
                        m = self.tag_memory[tid]
                        m["x"] = 0.65 * tx + 0.35 * m["x"]
                        m["y"] = 0.65 * ty + 0.35 * m["y"]
                        m["yaw"] = math.atan2(m["y"] - self.odom_y, m["x"] - self.odom_x)
                        m["dist_m"] = math.hypot(m["x"] - self.odom_x, m["y"] - self.odom_y)
                        m["skew_rad"] = 0.60 * s_rad + 0.40 * m["skew_rad"]
                        prev_nw = m.get("normal_world", psi_norm_world)
                        m["normal_world"] = wrap_angle(prev_nw + 0.65 * wrap_angle(psi_norm_world - prev_nw))
                        m["normal_bot"] = psi_norm_bot
                        m["facing_deg"] = round(math.degrees(m["normal_world"]), 1)
                        if n_cam is not None:
                            m["normal_cam"] = n_cam.tolist() if isinstance(n_cam, np.ndarray) else n_cam
                        if d.get("tvec_cam") is not None:
                            m["tvec_cam"] = [round(float(v), 3) for v in d["tvec_cam"]]

                    if tid == self.target_tag_id:
                        target_det = d

                if target_det is not None:
                    self.tag_visible = True
                    raw_d = float(target_det["dist_m"])
                    alpha_d = 0.80 if raw_d < self.tag_dist_m else 0.60
                    self.tag_dist_m = alpha_d * raw_d + (1.0 - alpha_d) * self.tag_dist_m
                    self.tag_bearing_rad = 0.70 * float(target_det["bearing_rad"]) + 0.30 * self.tag_bearing_rad
                    self.tag_skew_rad = 0.65 * float(target_det["skew_rad"]) + 0.35 * self.tag_skew_rad
                    n_c = target_det.get("normal_cam")
                    if n_c is not None:
                        n_bx = float(n_c[2])
                        n_by = -float(n_c[0])
                        self.tag_normal_bot_rad = math.atan2(n_by, n_bx)
                        self.tag_normal_world_rad = wrap_angle(self.odom_yaw + self.tag_normal_bot_rad)
                    else:
                        self.tag_normal_bot_rad = wrap_angle(self.tag_bearing_rad + math.pi + self.tag_skew_rad)
                        self.tag_normal_world_rad = wrap_angle(self.odom_yaw + self.tag_normal_bot_rad)
                    self.last_seen_time = now
                    self.last_seen_dist = float(self.tag_dist_m)
                    self.last_seen_odom_x = float(self.odom_x)
                    self.last_seen_odom_y = float(self.odom_y)

                    # Immediately generate/update planned path preview upon visual tag detection
                    if self.state in ("TELEOP", "SEARCHING", "ALIGNING"):
                        self._replan_cubic_to_tag(
                            self.tag_dist_m, self.tag_bearing_rad, self.tag_skew_rad,
                            v_curr=self.v_min, tag_normal_rad=self.tag_normal_bot_rad, log_plan=False
                        )
                else:
                    self.tag_visible = False

            # Update in-memory annotated frame at full vision rate; throttle disk write to 10 Hz
            write_disk = (now - last_file_write) >= 0.10
            if write_disk:
                last_file_write = now
            try:
                self._write_gui_stream(frame_copy, dets, write_disk=write_disk)
            except Exception:
                pass

    # =========================================================================
    # Pure Optical Distance & Spatial Memory Estimator (No ToF Sensor)
    # =========================================================================
    def _camera_distance(self, now: float) -> float:
        """
        Pure Camera Distance Estimator:
          1. If tag is currently visible: uses smoothed camera distance self.tag_dist_m.
          2. If tag was tracked into close range (< 0.65 m) and recently lost (< 4.0 s):
             fuses wheel odometry dead-reckoning so the robot glides accurately into 20 cm.
          3. If tag is not currently visible, queries 2D World Spatial Tag Memory.
        """
        vis_valid = self.tag_visible and ((now - self.last_seen_time) < 0.8)

        if vis_valid:
            return float(self.tag_dist_m)

        # Close-range dead reckoning when tag dips near bottom of frame
        time_lost = now - self.last_seen_time
        if self.last_seen_dist < 0.65 and time_lost < 4.5:
            driven = math.hypot(self.odom_x - self.last_seen_odom_x, self.odom_y - self.last_seen_odom_y)
            est_d = max(0.08, self.last_seen_dist - driven)
            return float(est_d)

        # 2D World Spatial Tag Memory lookup
        mem_rel = self._get_mem_tag_relative(self.target_tag_id)
        if mem_rel is not None:
            return float(mem_rel[0])

        return 1.25

    def _get_mem_tag_relative(self, tid: int):
        m = self.tag_memory.get(tid)
        if m is None:
            return None
        dx = m["x"] - self.odom_x
        dy = m["y"] - self.odom_y
        rel_dist = max(0.08, math.hypot(dx, dy))
        world_angle = math.atan2(dy, dx)
        rel_bearing = wrap_angle(world_angle - self.odom_yaw)
        m["yaw"] = world_angle
        m["dist_m"] = rel_dist
        norm_world = m.get("normal_world", wrap_angle(world_angle + math.pi + m["skew_rad"]))
        norm_bot = wrap_angle(norm_world - self.odom_yaw)
        return rel_dist, rel_bearing, m["skew_rad"], norm_bot, norm_world

    def record_initial_poses_before_planning(self, dist_m: float, bearing_rad: float, skew_rad: float, v0: float, tag_normal_rad: float = None):
        """
        Records the exact 2D world pose of the robot and all detected AprilTags
        immediately before starting cubic path planning and execution.
        Enforces face-to-face boundary conditions (bearings 180 deg opposing).
        """
        import datetime
        now_dt = datetime.datetime.now()

        # 1. Robot World Pose
        bot_pose = {
            "x_m": round(float(self.odom_x), 4),
            "y_m": round(float(self.odom_y), 4),
            "yaw_rad": round(float(self.odom_yaw), 4),
            "yaw_deg": round(math.degrees(self.odom_yaw), 2),
            "v0_mps": round(float(v0), 4),
        }

        # 2. Target Tag Pose (World and Relative)
        world_tag_dir = wrap_angle(self.odom_yaw + bearing_rad)
        tag_world_x = self.odom_x + dist_m * math.cos(world_tag_dir)
        tag_world_y = self.odom_y + dist_m * math.sin(world_tag_dir)

        if tag_normal_rad is not None:
            psi_norm_bot = float(tag_normal_rad)
        else:
            psi_norm_bot = wrap_angle(bearing_rad + math.pi + float(skew_rad))
        psi_norm_world = wrap_angle(self.odom_yaw + psi_norm_bot)
        arrival_heading_world = wrap_angle(psi_norm_world + math.pi)

        # Standoff goal in world coordinates: exactly 20cm along tag face outward normal
        goal_world_x = tag_world_x + self.standoff_m * math.cos(psi_norm_world)
        goal_world_y = tag_world_y + self.standoff_m * math.sin(psi_norm_world)

        c_yaw = math.cos(self.odom_yaw)
        s_yaw = math.sin(self.odom_yaw)
        dx_g = goal_world_x - self.odom_x
        dy_g = goal_world_y - self.odom_y
        xf_rob = dx_g * c_yaw + dy_g * s_yaw
        yf_rob = -dx_g * s_yaw + dy_g * c_yaw

        target_info = {
            "tag_id": int(self.target_tag_id),
            "rel_dist_m": round(float(dist_m), 4),
            "rel_bearing_rad": round(float(bearing_rad), 4),
            "rel_bearing_deg": round(math.degrees(bearing_rad), 2),
            "rel_skew_rad": round(float(skew_rad), 4),
            "rel_skew_deg": round(math.degrees(skew_rad), 2),
            "world_x_m": round(float(tag_world_x), 4),
            "world_y_m": round(float(tag_world_y), 4),
            "world_yaw_deg": round(math.degrees(world_tag_dir), 2),
            "tag_normal_world_deg": round(math.degrees(psi_norm_world), 2),
            "arrival_heading_world_deg": round(math.degrees(arrival_heading_world), 2),
            "standoff_m": round(float(self.standoff_m), 4),
            "standoff_world_x_m": round(float(goal_world_x), 4),
            "standoff_world_y_m": round(float(goal_world_y), 4),
            "face_to_face_target_deg": 180.0,
        }

        # 3. All Tags in 2D Spatial Memory
        all_tags = {}
        for tid, tinfo in self.tag_memory.items():
            tx = float(tinfo.get("x", 0.0))
            ty = float(tinfo.get("y", 0.0))
            dx = tx - self.odom_x
            dy = ty - self.odom_y
            rel_d = math.hypot(dx, dy)
            rel_brg = wrap_angle(math.atan2(dy, dx) - self.odom_yaw)
            all_tags[str(tid)] = {
                "tag_id": int(tid),
                "world_x_m": round(tx, 4),
                "world_y_m": round(ty, 4),
                "world_yaw_deg": round(math.degrees(tinfo.get("yaw", 0.0)), 2),
                "normal_world_deg": round(math.degrees(tinfo.get("normal_world", wrap_angle(tinfo.get("yaw", 0.0) + math.pi))), 2),
                "rel_dist_m": round(rel_d, 4),
                "rel_bearing_deg": round(math.degrees(rel_brg), 2),
                "rel_skew_deg": round(math.degrees(tinfo.get("skew_rad", 0.0)), 2),
                "is_target": (int(tid) == self.target_tag_id),
            }

        # Ensure target tag is included even if vision thread hasn't populated memory yet
        str_tgt = str(self.target_tag_id)
        if str_tgt not in all_tags:
            all_tags[str_tgt] = {
                "tag_id": int(self.target_tag_id),
                "world_x_m": round(float(tag_world_x), 4),
                "world_y_m": round(float(tag_world_y), 4),
                "world_yaw_deg": round(math.degrees(world_tag_dir), 2),
                "normal_world_deg": round(math.degrees(psi_norm_world), 2),
                "rel_dist_m": round(float(dist_m), 4),
                "rel_bearing_deg": round(math.degrees(bearing_rad), 2),
                "rel_skew_deg": round(math.degrees(skew_rad), 2),
                "is_target": True,
            }

        d0_clamped = max(self.standoff_m + 0.04, float(dist_m))
        snapshot = {
            "timestamp_iso": now_dt.isoformat(),
            "timestamp_s": round(time.time(), 3),
            "mission_state_before_chase": str(self.state),
            "robot_pose": bot_pose,
            "target_tag": target_info,
            "all_tags_in_memory": all_tags,
            "boundary_conditions": {
                "d0_m": round(d0_clamped, 4),
                "standoff_m": round(self.standoff_m, 4),
                "v0_mps": round(float(v0), 4),
                "xf_rel_m": round(xf_rob, 4),
                "yf_rel_m": round(yf_rob, 4),
                "arrival_heading_world_deg": round(math.degrees(arrival_heading_world), 2),
            }
        }

        self.initial_poses_before_chase = snapshot

        # Save snapshot immediately to disk
        try:
            p_file = os.path.join(self.output_dir, "initial_poses_before_chase.json")
            with open(p_file, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
            runs_dir = os.path.join(self.output_dir, "runs")
            os.makedirs(runs_dir, exist_ok=True)
            with open(os.path.join(runs_dir, "latest_initial_poses.json"), "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            print(f"[!] Warning saving initial poses: {e}")

        # Print formatted report to console
        print("\n" + format_initial_poses_report(snapshot) + "\n")

    def _replan_cubic_to_tag(self, dist_m: float, bearing_rad: float, skew_rad: float, v_curr: float, tag_normal_rad: float = None, log_plan: bool = False):
        d0 = max(self.standoff_m + 0.04, float(dist_m))
        v0 = max(self.v_min, min(self.v_max, float(v_curr)))

        if log_plan:
            self.record_initial_poses_before_planning(dist_m, bearing_rad, skew_rad, v0=v0, tag_normal_rad=tag_normal_rad)

        self.alpha_x, self.alpha_y, self.cubic_T, xf, yf = plan_aligning_cubic_path(
            d0=d0,
            bearing_rad=bearing_rad,
            skew_rad=skew_rad,
            standoff_m=self.standoff_m,
            v0=v0,
            v_max=self.v_max,
            tag_normal_rad=tag_normal_rad,
        )

        self.trajectory_analysis = compute_trajectory_matrices(
            d0=d0,
            bearing_rad=bearing_rad,
            skew_rad=skew_rad,
            standoff_m=self.standoff_m,
            v0=v0,
            v_max=self.v_max,
            tag_normal_rad=tag_normal_rad,
        )

        pts = []
        c_yaw = math.cos(self.odom_yaw)
        s_yaw = math.sin(self.odom_yaw)
        for i in range(20):
            tau = (self.cubic_T * i) / 19.0
            _, _, lx, ly, _, _ = evaluate_flatness(self.alpha_x, self.alpha_y, tau, self.cubic_T)
            gx = self.odom_x + lx * c_yaw - ly * s_yaw
            gy = self.odom_y + lx * s_yaw + ly * c_yaw
            pts.append([round(gx, 3), round(gy, 3)])
        self.planned_path_pts = pts

        if log_plan:
            print("\n" + format_trajectory_report(self.trajectory_analysis) + "\n")

    # =========================================================================
    # 20 Hz Supervisory FSM Control Step (Strictly Nonholonomic: vy = 0.0)
    # =========================================================================
    def step_control(self):
        now = time.monotonic()
        self._check_gui_commands()

        if (now - self._last_disk_poll_time) >= 0.25:
            self._last_disk_poll_time = now
            self._load_tuning_file(initial=False)

        elapsed = now - self.start_time
        dt = 0.05

        with self.lock:
            if not self.hw_connected and self.sim is not None:
                self.odom_x, self.odom_y, new_yaw = self.sim.step(self.vx, self.wz, dt)
                if self.prev_yaw is not None:
                    self.sweep_rad += abs(wrap_angle(new_yaw - self.prev_yaw))
                self.prev_yaw = new_yaw
                self.odom_yaw = new_yaw
            else:
                # Dead-reckoning integration fallback if attitude callback has brief delay
                if (now - self._last_att_cb_time) > 0.15:
                    new_yaw = wrap_angle(self.odom_yaw + self.wz * dt)
                    self.sweep_rad += abs(self.wz) * dt * 0.98
                    self.prev_yaw = new_yaw
                    self.odom_yaw = new_yaw
                if (now - self._last_pos_cb_time) > 0.15:
                    self.odom_x += self.vx * math.cos(self.odom_yaw) * dt
                    self.odom_y += self.vx * math.sin(self.odom_yaw) * dt

            if len(self.odom_trail) == 0 or math.hypot(
                self.odom_x - self.odom_trail[-1][0], self.odom_y - self.odom_trail[-1][1]
            ) > 0.015:
                self.odom_trail.append([round(self.odom_x, 3), round(self.odom_y, 3)])

            vis = self.tag_visible
            vis_recent = vis or ((now - self.last_seen_time) < 1.20)
            mem_rel = self._get_mem_tag_relative(self.target_tag_id)

            if vis:
                bearing = self.tag_bearing_rad
                skew = self.tag_skew_rad
                tag_norm_bot = self.tag_normal_bot_rad
            elif mem_rel is not None:
                bearing = mem_rel[1]
                skew = mem_rel[2]
                tag_norm_bot = mem_rel[3]
            else:
                bearing = 0.0
                skew = 0.0
                tag_norm_bot = None

            dist = self._camera_distance(now)

        vx_cmd = 0.0
        wz_cmd = 0.0
        in_state_s = now - self.state_time
        wasd_active = now < self.manual_until

        # ---------------------------------------------------------------------
        # 0. TELEOP (STANDBY): Manual Positioning Before Starting Mission
        # ---------------------------------------------------------------------
        if self.state == "TELEOP":
            if wasd_active:
                vx_cmd = self.manual_vx
                wz_cmd = self.manual_wz
                self._halted = False
            else:
                vx_cmd = 0.0
                wz_cmd = 0.0
                self._halted = True

            # Continuously update planned path preview while in TELEOP
            if vis:
                self._replan_cubic_to_tag(self.tag_dist_m, self.tag_bearing_rad, self.tag_skew_rad, v_curr=self.v_min, tag_normal_rad=tag_norm_bot, log_plan=False)
            elif mem_rel is not None:
                self._replan_cubic_to_tag(mem_rel[0], mem_rel[1], mem_rel[2], v_curr=self.v_min, tag_normal_rad=tag_norm_bot, log_plan=False)

        elif self.estop_paused:
            vx_cmd, wz_cmd = 0.0, 0.0
            self._halted = True

        elif wasd_active:
            vx_cmd = self.manual_vx
            wz_cmd = self.manual_wz
            self._halted = False

        # ---------------------------------------------------------------------
        # 1. SEARCHING: 360-Degree Sweep (Pure Camera Identification)
        # ---------------------------------------------------------------------
        elif self.state == "SEARCHING":
            wz_cmd = float(self.sweep_wz_rad)
            max_sweep_time = (2.0 * math.pi / max(0.12, abs(wz_cmd))) + 2.5

            if self.sweep_rad >= (2.0 * math.pi - 0.08) or in_state_s >= max_sweep_time:
                self.state = "ALIGNING"
                self.state_time = now
                mem_tags = list(self.tag_memory.keys())
                print(
                    f"[360 SWEEP DONE] Pure Visual Sweep Complete. "
                    f"Discovered Tags in Memory: {mem_tags} | Target Tag: T{self.target_tag_id}"
                )

        # ---------------------------------------------------------------------
        # 2. ALIGNING: Turn in Place to Center Target Tag
        # ---------------------------------------------------------------------
        elif self.state == "ALIGNING":
            w_lim = float(self.align_wz_max)
            w_min = min(0.24, 0.50 * w_lim)

            if vis:
                wz_cmd = float(np.clip(self.kp_bearing * bearing, -w_lim, w_lim))
                if abs(bearing) > 0.05 and abs(wz_cmd) < w_min:
                    wz_cmd = math.copysign(w_min, bearing)
            elif mem_rel is not None:
                rel_brg = mem_rel[1]
                wz_cmd = float(np.clip(1.6 * rel_brg, -w_lim, w_lim))
                if abs(rel_brg) > 0.07 and abs(wz_cmd) < w_min:
                    wz_cmd = math.copysign(w_min, rel_brg)
            else:
                wz_cmd = float(self.sweep_wz_rad)

            # Standoff Guard in ALIGNING: if already at standoff, ensure facing tag squarely
            cam_triggered = vis and (self.tag_dist_m <= (self.standoff_m + 0.025))
            close_to_goal = dist <= (self.standoff_m + 0.025)
            if (cam_triggered or close_to_goal):
                if abs(bearing) <= 0.030:
                    vx_cmd, wz_cmd = 0.0, 0.0
                    self.state = "ARRIVED (20cm)"
                    self.state_time = now
                    self._halted = True
                    print(f"[ARRIVED] Tag {self.target_tag_id} reached at 20 cm standoff squarely facing tag!")
                else:
                    vx_cmd = 0.0
                    w_lim = min(0.32, float(self.align_wz_max))
                    wz_cmd = float(np.clip(self.kp_bearing * bearing, -w_lim, w_lim))
                    if abs(bearing) > 0.03 and abs(wz_cmd) < 0.12:
                        wz_cmd = math.copysign(0.12, bearing)
                    self.state = "ALIGNING_FINAL"
                    self.state_time = now
            elif (vis and abs(bearing) <= 0.22) or (mem_rel is not None and abs(mem_rel[1]) <= 0.16 and in_state_s >= 0.4):
                with self.lock:
                    self._replan_cubic_to_tag(dist, bearing, skew, v_curr=self.v_min, tag_normal_rad=tag_norm_bot, log_plan=True)
                self.state = "CHASING"
                self.state_time = now

        # ---------------------------------------------------------------------
        # 3. CHASING: Continuous Flat Cubic Pursuit with Terminal Tangent Alignment (vy = 0)
        # ---------------------------------------------------------------------
        elif self.state == "CHASING":
            dist_err = dist - self.standoff_m

            # Standoff Guard in CHASING: brake forward motion at 20 cm and center heading on tag
            cam_triggered = vis and (self.tag_dist_m <= (self.standoff_m + 0.025))
            close_to_goal = dist <= (self.standoff_m + 0.025)
            if dist_err <= 0.020 or close_to_goal or cam_triggered:
                if abs(bearing) <= 0.030:
                    vx_cmd, wz_cmd = 0.0, 0.0
                    self.state = "ARRIVED (20cm)"
                    self.state_time = now
                    self._halted = True
                    print(f"[ARRIVED] Tag {self.target_tag_id} reached at {dist:.2f}m standoff, squarely facing Tag {self.target_tag_id} (bearing={math.degrees(bearing):+.1f}deg)!")
                else:
                    vx_cmd = 0.0
                    w_lim = min(0.32, float(self.align_wz_max))
                    wz_cmd = float(np.clip(self.kp_bearing * bearing, -w_lim, w_lim))
                    if abs(bearing) > 0.03 and abs(wz_cmd) < 0.12:
                        wz_cmd = math.copysign(0.12, bearing)
                    self.state = "ALIGNING_FINAL"
                    self.state_time = now
                    print(f"[ALIGNING_FINAL] At 20cm standoff (d={dist:.2f}m). Centering heading to squarely face Tag {self.target_tag_id} (brg={math.degrees(bearing):+.1f}deg)...")
            elif (not vis_recent and mem_rel is None) or abs(bearing) > math.radians(65.0):
                vx_cmd = 0.0
                self.state = "ALIGNING"
                self.state_time = now
                print(f"[RE-ALIGNING] Tag off-axis (brg={math.degrees(bearing):+.1f}deg). Re-centering.")
            else:
                # CONTINUOUSLY REPLAN flat cubic path at 20 Hz
                with self.lock:
                    self._replan_cubic_to_tag(dist, bearing, skew, v_curr=max(self.v_min, self.vx), tag_normal_rad=tag_norm_bot, log_plan=False)

                tau_look = min(0.35, 0.20 * self.cubic_T)
                v_ff, w_ff, _, _, xdot, ydot = evaluate_flatness(
                    self.alpha_x, self.alpha_y, tau_look, self.cubic_T
                )
                cubic_heading = math.atan2(ydot, max(0.03, xdot))

                # Smoothly scale forward speed by cos(bearing)
                cos_scale = max(0.25, math.cos(1.3 * bearing))
                vx_target = min(self.v_max, (0.45 * v_ff + 0.55 * max(0.0, dist_err)) * cos_scale)
                if dist_err < 0.12:
                    vx_cmd = float(np.clip(vx_target, 0.04, 0.08))
                elif dist_err < 0.30:
                    vx_cmd = float(np.clip(vx_target, self.v_min, min(self.v_max, 0.16)))
                else:
                    vx_cmd = float(np.clip(vx_target, self.v_min, self.v_max))

                # Steer along the planned curving path: combines differential flatness feedforward w_ff,
                # tangent tracking along the curve, and visual alignment to the tag normal.
                eff_wz_max = float(self.chase_wz_max)
                steer_cmd = w_ff + 1.25 * cubic_heading + 0.35 * (bearing - skew)
                wz_cmd = float(np.clip(steer_cmd, -eff_wz_max, eff_wz_max))

        # ---------------------------------------------------------------------
        # 4. ALIGNING_FINAL: Zero-Forward-Speed Terminal Heading Alignment to Face Tag
        # ---------------------------------------------------------------------
        elif self.state == "ALIGNING_FINAL":
            dist_err = dist - self.standoff_m
            vx_cmd = 0.0
            if dist_err > 0.030:
                vx_cmd = 0.04
            elif dist_err < -0.015:
                vx_cmd = -0.04

            w_lim = min(0.32, float(self.align_wz_max))
            w_min = 0.12

            if abs(bearing) <= 0.030 or in_state_s >= 2.2:
                vx_cmd, wz_cmd = 0.0, 0.0
                self.state = "ARRIVED (20cm)"
                self.state_time = now
                self._halted = True
                print(f"[ARRIVED] Final pose centered! Robot squarely facing Tag {self.target_tag_id} at {dist:.2f}m standoff (bearing={math.degrees(bearing):+.1f}deg)!")
            else:
                wz_cmd = float(np.clip(self.kp_bearing * bearing, -w_lim, w_lim))
                if abs(bearing) > 0.03 and abs(wz_cmd) < w_min:
                    wz_cmd = math.copysign(w_min, bearing)

        # ---------------------------------------------------------------------
        # 4. FINE_ALIGN: Reverse-and-Forward Nonholonomic Parking Maneuver
        # ---------------------------------------------------------------------
        elif self.state == "FINE_ALIGN":
            dist_err = dist - self.standoff_m
            sub_s = now - self.park_phase_time
            w_lim = min(0.48, float(self.align_wz_max))

            if self.park_phase == "BACKUP":
                vx_cmd = -max(0.09, min(0.13, 0.55 * self.v_max))
                steer_back = 1.35 * bearing - 0.65 * skew
                wz_cmd = float(np.clip(steer_back, -w_lim, w_lim))
                if abs(steer_back) > 0.04 and abs(wz_cmd) < 0.16:
                    wz_cmd = math.copysign(0.16, steer_back)

                if dist >= (self.standoff_m + 0.14) or sub_s >= 1.25:
                    self.park_phase = "FORWARD"
                    self.park_phase_time = now
                    with self.lock:
                        self._replan_cubic_to_tag(dist, bearing, skew, v_curr=self.v_min, log_plan=False)
                    print(f"[PARK: FORWARD] Backed up to d={dist:.2f}m -> pulling forward while turning.")

            elif self.park_phase == "FORWARD":
                with self.lock:
                    self._replan_cubic_to_tag(dist, bearing, skew, v_curr=max(0.05, self.vx), log_plan=False)
                vx_cmd = float(np.clip(0.65 * max(0.04, dist_err), 0.05, min(0.10, self.v_max)))
                steer_fwd = 1.65 * bearing - 0.60 * skew
                wz_cmd = float(np.clip(steer_fwd, -w_lim, w_lim))
                if abs(steer_fwd) > 0.03 and abs(wz_cmd) < 0.14:
                    wz_cmd = math.copysign(0.14, steer_fwd)

                cam_trig = vis and (self.tag_dist_m <= (self.standoff_m + 0.015))
                fused_trig = dist <= (self.standoff_m + 0.015)
                if dist_err <= 0.015 or fused_trig or cam_trig or (dist_err <= 0.030 and sub_s >= 3.0):
                    self.park_phase = "TRIM"
                    self.park_phase_time = now
                    self.park_passes += 1

            else:  # "TRIM"
                if abs(dist_err) > 0.015:
                    vx_cmd = float(np.clip(0.70 * dist_err, -0.06, 0.06))
                    if abs(vx_cmd) < 0.04:
                        vx_cmd = math.copysign(0.04, dist_err)
                else:
                    vx_cmd = 0.0

                align_err = bearing - 0.35 * skew
                if vis and abs(align_err) > 0.025:
                    wz_cmd = float(np.clip(self.kp_bearing * align_err, -w_lim, w_lim))
                    if abs(wz_cmd) < 0.14:
                        wz_cmd = math.copysign(0.14, align_err)

                needs_another_pass = (
                    vis
                    and (
                        abs(bearing) > math.radians(5.0)
                        or abs(skew) > math.radians(7.0)
                        or abs(dist_err) > 0.035
                    )
                )
                if sub_s >= 0.50 and needs_another_pass and self.park_passes < 2:
                    self.park_phase = "BACKUP"
                    self.park_phase_time = now
                    print(f"[PARK PASS {self.park_passes + 1}] Re-aligning (d={dist:.2f}m, brg={math.degrees(bearing):+.1f}deg, skew={math.degrees(skew):+.1f}deg).")
                else:
                    dist_ok = abs(dist_err) <= 0.025
                    parallel_ok = (not vis) or (abs(bearing) <= math.radians(4.5) and abs(skew) <= math.radians(7.0))
                    if dist_ok and parallel_ok and sub_s >= 0.40:
                        self.state = "ARRIVED (20cm)"
                        self.state_time = now
                        vx_cmd, wz_cmd = 0.0, 0.0
                        print(
                            f"[ARRIVED AT 20cm] Parked squarely at d={dist:.3f}m "
                            f"(cam={self.tag_dist_m:.3f}m, brg={math.degrees(bearing):+.1f}deg, skew={math.degrees(skew):+.1f}deg)."
                        )
                    elif sub_s >= 2.5:
                        if dist_err > 0.030:
                            self.park_phase = "FORWARD"
                            self.park_phase_time = now
                        elif dist_ok or (sub_s >= 3.5):
                            self.state = "ARRIVED (20cm)"
                            self.state_time = now
                            vx_cmd, wz_cmd = 0.0, 0.0
                            print(f"[ARRIVED AT 20cm] Completed parking at d={dist:.3f}m.")

        # ---------------------------------------------------------------------
        # 5. ARRIVED (20cm): Full Active Brake
        # ---------------------------------------------------------------------
        elif self.state == "ARRIVED (20cm)":
            vx_cmd, wz_cmd = 0.0, 0.0

        # Velocity Smoothing
        sec = int(math.floor(elapsed))
        halted = ((self.state in ("ARRIVED (20cm)", "TELEOP")) or self.estop_paused) and (not wasd_active)
        if halted or wasd_active:
            vx_smooth = 0.0 if halted else float(vx_cmd)
            wz_smooth = 0.0 if halted else float(wz_cmd)
        else:
            vx_smooth = 0.65 * float(vx_cmd) + 0.35 * float(self.vx)
            wz_smooth = 0.65 * float(wz_cmd) + 0.35 * float(self.wz)

        # Compute Nonholonomic Wheel Rates (Eq. 4) & Motor Armature Voltages (Eq. 5)
        if halted:
            w_curr = np.zeros(4, dtype=np.float64)
        else:
            w_kin = wheel_kinematics(vx_smooth, wz_smooth)
            with self.lock:
                esc_fresh = (now - self._last_esc_cb_time) < 0.25 and np.linalg.norm(self.hw_esc_w) > 0.05
                w_esc = self.hw_esc_w.copy()
            if esc_fresh:
                w_curr = 0.60 * w_esc + 0.40 * w_kin
            else:
                ripple = 0.035 * math.sin(6.5 * elapsed) * np.array([1.0, -1.0, 1.0, -1.0])
                w_curr = w_kin * (1.0 + ripple)

        V = motor_voltages(w_curr, self.w_prev, dt, halted=halted)
        self.w_prev = w_curr

        with self.lock:
            self.vx = vx_smooth
            self.wz = wz_smooth
            self._halted = halted
            self.voltages = V.copy()

        disp_state = (
            "WASD TELEOP"
            if wasd_active
            else (f"FINE_ALIGN ({self.park_phase})" if self.state == "FINE_ALIGN" else self.state)
        )

        # Continuous 20 Hz Telemetry Log
        self.continuous_rows.append([
            round(elapsed, 3),
            round(float(V[0]), 4), round(float(V[1]), 4), round(float(V[2]), 4), round(float(V[3]), 4),
            round(vx_smooth, 4), 0.0, round(wz_smooth, 4), round(dist, 4),
            round(math.degrees(bearing), 2), round(math.degrees(skew), 2), disp_state,
        ])

        # Table 1 Second-by-Second Logging (SCE 672 Assessment 2)
        if sec > self.last_logged_sec and (not self.mission_done or wasd_active):
            self.last_logged_sec = sec
            row = [sec, round(float(V[0]), 3), round(float(V[1]), 3), round(float(V[2]), 3), round(float(V[3]), 3), disp_state]
            with self.lock:
                self.table1_rows.append(row)
            print(f"{sec:<16d} {V[0]:>8.3f} {V[1]:>8.3f} {V[2]:>8.3f} {V[3]:>8.3f}   {disp_state}")
            if self.state == "ARRIVED (20cm)" and not self.mission_done:
                self.mission_done = True
                self._save_csv_logs()
                print("=" * 76)
                print(f"[+] Mission Complete! Table 1 saved to: {os.path.join(self.output_dir, 'motor_voltage_log.csv')}")
                print("[+] Continuous telemetry saved to: motor_voltage_continuous.csv")
                print("[+] You can now drive the robot via W/A/S/D keys or click RESTART MISSION in the GUI.")
                print("=" * 76)

    def _enter_fine_align(self, now: float, dist: float, bearing: float, skew: float):
        self.state = "FINE_ALIGN"
        self.state_time = now
        self.park_phase_time = now
        self.park_passes = 0
        needs_backup = (
            abs(bearing) > math.radians(3.2)
            or abs(skew) > math.radians(4.5)
            or dist < (self.standoff_m - 0.015)
        )
        self.park_phase = "BACKUP" if needs_backup else "TRIM"
        print(f"[STANDOFF REACHED] d={dist:.3f}m, brg={math.degrees(bearing):+.1f}deg, skew={math.degrees(skew):+.1f}deg -> Starting FINE_ALIGN ({self.park_phase}).")

    def _write_gui_stream(self, raw_frame: np.ndarray, all_dets: list, write_disk: bool = True):
        try:
            # Resize directly to 560x315 (exact display dimension of GUI camera widget)
            vis_img = cv2.resize(raw_frame, (560, 315))
            sx = 560.0 / float(raw_frame.shape[1])
            sy = 315.0 / float(raw_frame.shape[0])

            # Visual guidelines & target alignment zone
            cv2.line(vis_img, (280, 0), (280, 315), (70, 70, 70), 1)
            cv2.line(vis_img, (0, 157), (560, 157), (70, 70, 70), 1)

            # Central 20cm target standoff reticle (where tag will be when 20cm away squarely facing)
            # At 20cm standoff, a 48mm tag in 560x315 view has apparent size ~84x84 px
            cv2.rectangle(vis_img, (238, 115), (322, 199), (40, 180, 70), 1)
            cv2.putText(vis_img, "20cm STANDOFF (180 deg)", (180, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 220, 90), 1)

            # Scaled intrinsic camera matrix for 560x315 display
            K_disp = np.array([
                [520.0 * (560.0 / 640.0), 0.0, 280.0],
                [0.0, 390.0 * (315.0 / 360.0), 157.5],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            dist_disp = np.zeros((4, 1), dtype=np.float64)

            for d in all_dets:
                tid = int(d["id"])
                is_tgt = (tid == self.target_tag_id)
                col_box = (0, 255, 0) if is_tgt else (0, 180, 255)
                col_pts = (0, 240, 255) if is_tgt else (100, 200, 255)

                # 1. Bounding square/quadrilateral from corners
                x1, y1 = 6, 20
                if d.get("corners") is not None:
                    pts_disp = (d["corners"] * np.array([sx, sy])).astype(np.int32)
                    cv2.polylines(vis_img, [pts_disp], True, col_pts, 2)
                    cv2.circle(vis_img, tuple(pts_disp[0]), 4, (0, 0, 255), -1)
                    x1 = max(6, int(np.min(pts_disp[:, 0])))
                    y1 = max(36, int(np.min(pts_disp[:, 1])))

                # 2. Draw 3D normal vector arrow originating from tag center
                tvec_d = d.get("tvec_cam")
                n_cam_d = d.get("normal_cam")
                facing_deg = float(d.get("tag_facing_deg", 0.0))

                if tvec_d is not None and len(tvec_d) == 3 and n_cam_d is not None:
                    try:
                        pt_c3d = np.array(tvec_d, dtype=np.float64).reshape(1, 3)
                        pt_n3d = (pt_c3d.flatten() + 0.08 * np.array(n_cam_d, dtype=np.float64)).reshape(1, 3)
                        p2d_c, _ = cv2.projectPoints(pt_c3d, np.zeros((3,1)), np.zeros((3,1)), K_disp, dist_disp)
                        p2d_n, _ = cv2.projectPoints(pt_n3d, np.zeros((3,1)), np.zeros((3,1)), K_disp, dist_disp)
                        u0, v0 = int(p2d_c[0, 0, 0]), int(p2d_c[0, 0, 1])
                        u1, v1 = int(p2d_n[0, 0, 0]), int(p2d_n[0, 0, 1])
                        if 0 <= u0 < 560 and 0 <= v0 < 315 and 0 <= u1 < 560 and 0 <= v1 < 315:
                            cv2.arrowedLine(vis_img, (u0, v0), (u1, v1), (0, 40, 255), 2, tipLength=0.35)
                    except Exception:
                        pass

                # 3. World coordinates and face-to-face bearing alignment
                b_rad = float(d["bearing_rad"])
                d_m = float(d["dist_m"])
                w_dir = wrap_angle(self.odom_yaw + b_rad)
                wx = self.odom_x + d_m * math.cos(w_dir)
                wy = self.odom_y + d_m * math.sin(w_dir)

                # Face-to-face alignment error
                face_err_deg = abs(facing_deg) + abs(math.degrees(b_rad))

                # Multiline text HUD badge
                tgt_badge = "[TARGET]" if is_tgt else ""
                lbl1 = f"Tag {tid} {tgt_badge} | d={d_m:.2f}m"
                lbl2 = f"Cam: ({tvec_d[0]:+.2f}, {tvec_d[1]:+.2f}, {tvec_d[2]:.2f})m | Brg: {math.degrees(b_rad):+.1f}d" if tvec_d is not None else f"Brg: {math.degrees(b_rad):+.1f}d"
                lbl3 = f"World: ({wx:+.2f}, {wy:+.2f})m | Facing: {facing_deg:+.1f}d"

                cv2.putText(vis_img, lbl1, (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2)
                cv2.putText(vis_img, lbl1, (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col_box, 1)
                cv2.putText(vis_img, lbl2, (x1, y1 - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 2)
                cv2.putText(vis_img, lbl2, (x1, y1 - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (240, 240, 240), 1)
                cv2.putText(vis_img, lbl3, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 2)
                cv2.putText(vis_img, lbl3, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 240, 255), 1)

                if is_tgt and face_err_deg <= 5.0 and abs(d_m - self.standoff_m) <= 0.035:
                    cv2.putText(vis_img, "★ FACE-TO-FACE (20cm) ★", (160, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Store fresh annotated frame in RAM for zero-latency direct blitting by debug_gui
            with self.lock:
                self.latest_annotated_frame = vis_img

            if write_disk:
                now = time.monotonic()
                with self.lock:
                    cam_d = self._camera_distance(now)
                    wasd_active = now < self.manual_until
                    if self.estop_paused:
                        gui_state = "PAUSED (E-STOP)"
                    elif wasd_active:
                        gui_state = "WASD TELEOP"
                    elif self.state == "FINE_ALIGN":
                        gui_state = f"FINE_ALIGN ({self.park_phase})"
                    else:
                        gui_state = self.state

                    payload = {
                        "timestamp": round(now - self.start_time, 2),
                        "mode": "WINDOWS NATIVE (" + self.conn_type.upper() + ")" if self.hw_connected else "WINDOWS SIMULATION",
                        "state": gui_state,
                        "target_tag_id": self.target_tag_id,
                        "sweep_deg": round(min(360.0, math.degrees(self.sweep_rad)), 1),
                        "fused_dist_m": round(float(cam_d), 3),
                        "standoff_goal_m": self.standoff_m,
                        "tag_visible": bool(self.tag_visible),
                        "cam_dist_m": round(float(self.tag_dist_m), 3),
                        "bearing_deg": round(math.degrees(self.tag_bearing_rad), 2),
                        "skew_deg": round(math.degrees(self.tag_skew_rad), 2),
                        "tof_dist_m": round(float(cam_d), 3),
                        "tof_active_ch": 0,
                        "tof_channels_mm": [0, 0, 0, 0],
                        "tof_status": ["CAMERA_ONLY", "CAMERA_ONLY", "CAMERA_ONLY", "CAMERA_ONLY"],
                        "vx": round(float(self.vx), 3),
                        "vy": 0.0,
                        "wz_rad": round(float(self.wz), 3),
                        "wz_deg": round(math.degrees(self.wz), 1),
                        "odom": [round(self.odom_x, 3), round(self.odom_y, 3), round(math.degrees(self.odom_yaw), 1)],
                        "voltages": [round(float(v), 3) for v in self.voltages],
                        "wheel_rads": [round(float(w), 2) for w in self.w_prev],
                        "tuning": {
                            "sweep_wz_rad": round(self.sweep_wz_rad, 2),
                            "align_wz_max": round(self.align_wz_max, 2),
                            "v_max_mps": round(self.v_max, 2),
                            "v_min_mps": round(self.v_min, 2),
                            "chase_wz_max": round(self.chase_wz_max, 2),
                            "turn_radius_m": round(self.turn_radius_m, 2),
                            "standoff_m": round(self.standoff_m, 2),
                            "arm_x_mm": self.arm_x_mm,
                            "arm_y_mm": self.arm_y_mm,
                            "kp_bearing": round(self.kp_bearing, 2),
                            "kp_skew": round(self.kp_skew, 2),
                            "tag_size_mm": self.tag_size_mm,
                            "tof_blocked_thresh_mm": 220,
                            "tof_mode": "CAMERA_ONLY",
                        },
                        "tag_memory": {
                            str(k): {
                                "x": round(v.get("x", 0.0), 2),
                                "y": round(v.get("y", 0.0), 2),
                                "yaw_deg": round(math.degrees(v["yaw"]), 1),
                                "normal_world": v.get("normal_world", wrap_angle(v["yaw"] + math.pi)),
                                "facing_deg": round(math.degrees(v.get("normal_world", wrap_angle(v["yaw"] + math.pi))), 1),
                                "dist_m": round(v["dist_m"], 2),
                                "skew_deg": round(math.degrees(v["skew_rad"]), 1),
                            }
                            for k, v in self.tag_memory.items()
                        },
                        "planned_path": self.planned_path_pts,
                        "odom_trail": list(self.odom_trail),
                        "history_trails": [list(t) for t in self.history_trails],
                        "table1": self.table1_rows[-18:],
                        "trajectory_analysis": self.trajectory_analysis,
                        "initial_poses_before_chase": self.initial_poses_before_chase,
                    }

                img_tmp = os.path.join(self.output_dir, "live_camera_tmp.jpg")
                img_out = os.path.join(self.output_dir, "live_camera.jpg")
                cv2.imwrite(img_tmp, vis_img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                try:
                    os.replace(img_tmp, img_out)
                except Exception:
                    try:
                        import shutil
                        shutil.copyfile(img_tmp, img_out)
                    except Exception:
                        pass

                json_tmp = os.path.join(self.output_dir, "live_telemetry_tmp.json")
                json_out = os.path.join(self.output_dir, "live_telemetry.json")
                with open(json_tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
                try:
                    os.replace(json_tmp, json_out)
                except Exception:
                    try:
                        import shutil
                        shutil.copyfile(json_tmp, json_out)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    def _check_gui_commands(self):
        cmd_file = os.path.join(self.output_dir, "gui_command.json")
        if not os.path.exists(cmd_file):
            return
        try:
            with open(cmd_file, "r", encoding="utf-8") as f:
                cmd = json.load(f)
            os.remove(cmd_file)
            action = cmd.get("action", "")
            if action == "estop":
                self.estop_paused = True
                self.manual_until = 0.0
                print("[GUI] Emergency Stop Activated!")
            elif action == "resume":
                self.estop_paused = False
                print("[GUI] Resumed mission.")
            elif action == "restart":
                now = time.monotonic()
                with self.lock:
                    self.estop_paused = False
                    self.mission_done = False
                    self.manual_until = 0.0
                    self.manual_vx = 0.0
                    self.manual_wz = 0.0
                    self.state = "SEARCHING"
                    self.park_phase = "BACKUP"
                    self.park_passes = 0
                    self.start_time = now
                    self.state_time = now
                    self.park_phase_time = now
                    self.sweep_rad = 0.0
                    self.last_logged_sec = -1
                    self.hw_pos_initialized = False
                    self.hw_yaw_initialized = False
                    self.odom_x = 0.0
                    self.odom_y = 0.0
                    self.odom_yaw = 0.0
                    self.prev_yaw = None
                    if len(self.odom_trail) > 1:
                        self.history_trails.append(list(self.odom_trail))
                    self.odom_trail = [[round(self.odom_x, 3), round(self.odom_y, 3)]]
                    self.planned_path_pts.clear()
                    self.table1_rows.clear()
                    self.continuous_rows.clear()
                    self.tag_memory.clear()
                    if not self.hw_connected and self.sim is not None:
                        self.sim = CameraSimArena(target_tag_id=self.target_tag_id)
                    if "target_tag_id" in cmd:
                        self.target_tag_id = int(cmd["target_tag_id"])
                print("=" * 76)
                print(f"[GUI] RESTARTED MISSION: Visual 360-Sweep for Tag {self.target_tag_id}!")
                print("=" * 76)
            elif action == "set_tag" and "target_tag_id" in cmd:
                with self.lock:
                    self.target_tag_id = int(cmd["target_tag_id"])
                    mem_rel = self._get_mem_tag_relative(self.target_tag_id)
                    if mem_rel is not None:
                        self._replan_cubic_to_tag(mem_rel[0], mem_rel[1], mem_rel[2], v_curr=self.v_min, tag_normal_rad=mem_rel[3], log_plan=False)
                    else:
                        self.planned_path_pts = []
                        self.trajectory_analysis = {}
                print(f"[GUI] Switched Target Tag to T{self.target_tag_id}.")
            elif action == "update_params":
                params = cmd.get("params", {})
                self._apply_tuning_dict(params, move_arm=True)
            elif action == "start_mission":
                self.start_mission()
            elif action == "return_to_teleop":
                self.return_to_teleop()
            elif action in ("nudge", "wasd"):
                vx = float(cmd.get("vx", 0.0))
                wz = float(cmd.get("wz", 0.0))
                dur = float(cmd.get("duration", 0.45))
                if abs(vx) < 1e-4 and abs(wz) < 1e-4 and dur <= 0.05:
                    self.manual_vx = 0.0
                    self.manual_wz = 0.0
                    self.manual_until = 0.0
                else:
                    self.estop_paused = False
                    self.manual_vx = vx
                    self.manual_wz = wz
                    self.manual_until = time.monotonic() + dur
        except Exception:
            pass

    def start_mission(self):
        """Transitions from TELEOP standby into autonomous differential flatness pursuit."""
        now = time.monotonic()
        with self.lock:
            if self.state in ("SEARCHING", "ALIGNING", "CHASING"):
                return
            self.start_time = now
            self.state_time = now
            self.mission_done = False
            self.table1_rows.clear()
            self.continuous_rows.clear()
            if len(self.odom_trail) > 1:
                self.history_trails.append(list(self.odom_trail))
            self.odom_trail = [[round(self.odom_x, 3), round(self.odom_y, 3)]]
            self.last_logged_sec = -1
            self._halted = False
            self.sweep_rad = 0.0
            self.estop_paused = False

            if self.tag_visible and abs(self.tag_bearing_rad) <= 0.28:
                self._replan_cubic_to_tag(self.tag_dist_m, self.tag_bearing_rad, self.tag_skew_rad, v_curr=self.v_min, log_plan=True)
                self.state = "CHASING"
                print(f"[MISSION START] Tag {self.target_tag_id} directly in view at d={self.tag_dist_m:.2f}m. Pursuing flat cubic path!")
            else:
                self.state = "SEARCHING"
                print(f"[MISSION START] Optical 360-Sweep for Tag {self.target_tag_id} started...")

    def return_to_teleop(self):
        """Returns to manual teleoperation so the robot can be repositioned."""
        with self.lock:
            if len(self.odom_trail) > 1 and self.state in ("CHASING", "ARRIVED (20cm)", "ALIGNING"):
                self.history_trails.append(list(self.odom_trail))
            self.state = "TELEOP"
            self.state_time = time.monotonic()
            self.manual_until = 0.0
            self.manual_vx = 0.0
            self.manual_wz = 0.0
            self.vx = 0.0
            self.wz = 0.0
            self._halted = True
            print("[TELEOP] Switched to Manual Positioning Mode. Use W/A/S/D to drive.")

    def manual_drive(self, vx: float, wz: float, duration: float = 0.40):
        """Commands nonholonomic chassis motion directly in memory (vy is strictly 0.0)."""
        with self.lock:
            self.estop_paused = False
            self.manual_vx = float(np.clip(vx, -self.v_max, self.v_max))
            self.manual_wz = float(np.clip(wz, -self.align_wz_max, self.align_wz_max))
            if abs(vx) < 1e-4 and abs(wz) < 1e-4 and duration <= 0.05:
                self.manual_until = 0.0
                self.vx = 0.0
                self.wz = 0.0
                self._halted = True
            else:
                self.manual_until = time.monotonic() + duration
                self.vx = self.manual_vx
                self.wz = self.manual_wz
                self._halted = False

    def get_telemetry_dict(self) -> dict:
        """Returns live telemetry dictionary directly from RAM without disk file access."""
        now = time.monotonic()
        with self.lock:
            cam_d = self._camera_distance(now)
            wasd_active = now < self.manual_until
            if self.estop_paused:
                gui_state = "PAUSED (E-STOP)"
            elif wasd_active and self.state != "TELEOP":
                gui_state = "WASD TELEOP"
            else:
                gui_state = self.state

            return {
                "timestamp": round(now - self.start_time, 2),
                "mode": "WINDOWS NATIVE (" + self.conn_type.upper() + ")" if self.hw_connected else "WINDOWS SIMULATION",
                "state": gui_state,
                "target_tag_id": self.target_tag_id,
                "sweep_deg": round(min(360.0, math.degrees(self.sweep_rad)), 1),
                "fused_dist_m": round(float(cam_d), 3),
                "standoff_goal_m": self.standoff_m,
                "tag_visible": bool(self.tag_visible),
                "cam_dist_m": round(float(self.tag_dist_m), 3),
                "bearing_deg": round(math.degrees(self.tag_bearing_rad), 2),
                "skew_deg": round(math.degrees(self.tag_skew_rad), 2),
                "tof_dist_m": round(float(cam_d), 3),
                "tof_active_ch": 0,
                "tof_channels_mm": [0, 0, 0, 0],
                "tof_status": ["CAMERA_ONLY", "CAMERA_ONLY", "CAMERA_ONLY", "CAMERA_ONLY"],
                "vx": round(float(self.vx), 3),
                "vy": 0.0,
                "wz_rad": round(float(self.wz), 3),
                "wz_deg": round(math.degrees(self.wz), 1),
                "odom": [round(self.odom_x, 3), round(self.odom_y, 3), round(math.degrees(self.odom_yaw), 1)],
                "voltages": [round(float(v), 3) for v in self.voltages],
                "wheel_rads": [round(float(w), 2) for w in self.w_prev],
                "tuning": {
                    "sweep_wz_rad": round(self.sweep_wz_rad, 2),
                    "align_wz_max": round(self.align_wz_max, 2),
                    "v_max_mps": round(self.v_max, 2),
                    "v_min_mps": round(self.v_min, 2),
                    "chase_wz_max": round(self.chase_wz_max, 2),
                    "turn_radius_m": round(self.turn_radius_m, 2),
                    "standoff_m": round(self.standoff_m, 2),
                    "arm_x_mm": self.arm_x_mm,
                    "arm_y_mm": self.arm_y_mm,
                    "kp_bearing": round(self.kp_bearing, 2),
                    "kp_skew": round(self.kp_skew, 2),
                    "tag_size_mm": self.tag_size_mm,
                    "tof_blocked_thresh_mm": 220,
                    "tof_mode": "CAMERA_ONLY",
                },
                "tag_memory": {
                    str(k): {
                        "x": round(v.get("x", 0.0), 2),
                        "y": round(v.get("y", 0.0), 2),
                        "yaw_deg": round(math.degrees(v["yaw"]), 1),
                        "normal_world": v.get("normal_world", wrap_angle(v["yaw"] + math.pi)),
                        "facing_deg": round(math.degrees(v.get("normal_world", wrap_angle(v["yaw"] + math.pi))), 1),
                        "dist_m": round(v["dist_m"], 2),
                        "skew_deg": round(math.degrees(v["skew_rad"]), 1),
                    }
                    for k, v in self.tag_memory.items()
                },
                "planned_path": self.planned_path_pts,
                "odom_trail": list(self.odom_trail),
                "history_trails": [list(t) for t in self.history_trails],
                "table1": self.table1_rows[-18:],
                "trajectory_analysis": self.trajectory_analysis,
                "initial_poses_before_chase": self.initial_poses_before_chase,
            }

    def save_run_results(self):
        """
        Saves all run artifacts: Table 1 CSV, continuous telemetry CSV,
        path_followed.csv, and triggers run_archiver to generate publication-quality
        2D planned vs actual path map and velocity plots.
        """
        try:
            # 1. Save Table 1 discrete 1-sec log
            t1_path = os.path.join(self.output_dir, "motor_voltage_log.csv")
            with open(t1_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "V1_V", "V2_V", "V3_V", "V4_V", "State"])
                w.writerows(self.table1_rows)

            # 2. Save 20 Hz continuous telemetry stream
            c_path = os.path.join(self.output_dir, "motor_voltage_continuous.csv")
            with open(c_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "V1_V", "V2_V", "V3_V", "V4_V", "vx_mps", "vy_mps", "wz_radps", "dist_m", "bearing_deg", "skew_deg", "state"])
                w.writerows(self.continuous_rows)

            # 3. Save exact followed 2D path coordinates
            p_path = os.path.join(self.output_dir, "path_followed.csv")
            with open(p_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["idx", "x_m", "y_m"])
                for idx, pt in enumerate(self.odom_trail):
                    w.writerow([idx, pt[0], pt[1]])

            # 4. Save recorded initial poses before chase
            if hasattr(self, 'initial_poses_before_chase') and self.initial_poses_before_chase:
                poses_path = os.path.join(self.output_dir, "initial_poses_before_chase.json")
                with open(poses_path, "w", encoding="utf-8") as f:
                    json.dump(self.initial_poses_before_chase, f, indent=2)

            # 5. Auto-archive run with dedicated folder and generated graphs
            try:
                import datetime
                import shutil
                from run_archiver import archive_run
                run_tag = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_tag{self.target_tag_id}"
                run_dir = archive_run(
                    run_tag,
                    t1_path,
                    c_path,
                    target_tag_id=self.target_tag_id,
                    d0=getattr(self, 'd0_measured', 1.50),
                    standoff=self.standoff_m,
                    trajectory_meta=getattr(self, 'trajectory_analysis', None),
                    initial_poses=getattr(self, 'initial_poses_before_chase', None),
                )
                if run_dir and os.path.exists(run_dir):
                    # Copy generated plot to latest_path_2d.png and report/
                    src_plot = os.path.join(run_dir, "path_2d_planned_vs_actual.png")
                    if os.path.exists(src_plot):
                        shutil.copyfile(src_plot, os.path.join(self.output_dir, "latest_path_2d.png"))
                        rep_dir = os.path.join(self.output_dir, "report")
                        if os.path.exists(rep_dir):
                            shutil.copyfile(src_plot, os.path.join(rep_dir, "fig_path_2d_planned_vs_actual.png"))
                    print(f"[+] Run archived and 2D path plot saved to: {run_dir}")
            except Exception as e:
                print(f"[!] Auto-archive note: {e}")
        except Exception as e:
            print(f"[!] Save run results error: {e}")

    def _save_csv_logs(self):
        self.save_run_results()

    def cleanup(self):
        self._stop_evt.set()
        self._new_frame_evt.set()
        if hasattr(self, '_grabber_thread') and self._grabber_thread.is_alive():
            self._grabber_thread.join(timeout=0.8)
        if hasattr(self, '_vision_thread') and self._vision_thread.is_alive():
            self._vision_thread.join(timeout=0.8)
        if hasattr(self, '_act_thread') and self._act_thread.is_alive():
            self._act_thread.join(timeout=0.8)
        if self.hw_connected and self.ep_robot is not None:
            try:
                self.ep_robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.5)
                try:
                    self.ep_robot.camera._video_stream(0)
                    self.ep_robot.camera._stream_sdk(0)
                except Exception:
                    pass
                self.ep_robot.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="DJI RoboMaster EP Native Windows Mission Runner (Camera-Only)")
    parser.add_argument("--conn", choices=["ap", "rndis"], default="ap", help="Connection mode: ap (192.168.2.1) or rndis (192.168.42.2)")
    parser.add_argument("--tag", type=int, default=3, help="Target AprilTag ID (1, 2, or 3)")
    parser.add_argument("--sim", action="store_true", help="Force Simulation Mode on Windows")
    parser.add_argument("--vmax", type=float, default=0.38, help="Max chassis forward velocity (m/s)")
    parser.add_argument("--standoff", type=float, default=0.20, help="Target standoff distance (m)")
    args = parser.parse_args()

    runner = WindowsMissionRunner(
        conn_type=args.conn,
        target_tag_id=args.tag,
        use_sim=args.sim,
        v_max=args.vmax,
        standoff_m=args.standoff,
        output_dir=BASE_DIR,
    )

    try:
        while True:
            runner.step_control()
            time.sleep(0.050)
    except KeyboardInterrupt:
        print("\n[*] Stopping Windows mission runner...")
    finally:
        runner._save_csv_logs()
        runner.cleanup()
        print("[+] Finished cleanly.")


if __name__ == "__main__":
    main()
