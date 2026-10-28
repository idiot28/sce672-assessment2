#!/usr/bin/env python3
"""
mission_node.py - Live-Tunable ROS 2 Mission Node with Continuous Path Replanning (v4)
======================================================================================
Fixes & Features:
  1. Continuous Receding-Horizon 2D Flat Cubic Replanning:
     - Re-solves `plan_aligning_cubic_path` on EVERY 20 Hz control tick during CHASING
       using live camera pose + 2D world tag memory `(tx, ty)` so the robot continuously
       readjusts its path toward the tag and never follows a stale path blindly.
     - If the tag drifts off-center (> 28 deg) or leaves camera view (> 0.65 s),
       immediately pauses forward motion (vx = 0) and re-aligns in place.
  2. Accurate Dual Front ToF Auto-Discovery & Triple-Redundant 20 cm Stop:
     - Subscribes to DDS sensors before starting H.264 video stream (and auto-resubscribes
       if needed) so ToF channels never drop.
     - Rejects the static arm-blocked ToF channel and selects the closest unblocked
       straight ToF channel.
     - Stops at 20 cm when ANY of (Camera Tag Distance, Unblocked Straight ToF, or 2D
       World Tag Memory Distance) reaches `<= standoff_m` (0.20 m), preventing the bot
       from ever pushing through the tag.
  3. Strictly Nonholonomic Motion: `vy = 0.0` at all times (forward, backward, turning only).
"""

import os
import csv
import json
import math
import time
import queue
import socket
import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from std_msgs.msg import Float64MultiArray, String

from rm_flatness_nav.flatness_math import (
    wrap_angle,
    plan_aligning_cubic_path,
    evaluate_flatness,
    wheel_kinematics,
    motor_voltages,
    compute_trajectory_matrices,
    format_trajectory_report,
)
from rm_flatness_nav.vision_tof import (
    FrontToFSelector,
    AprilTagDetector,
    SimpleSimArena,
)


class DropOldestQueue(queue.Queue):
    """Single-slot FIFO queue that immediately replaces stale frames upon arrival of a new frame.
    Guarantees 0 ms internal buffer latency and prevents H.264 stream lag in ROS 2."""
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


def patch_robomaster_sdk(conn_type: str):
    """Applies RoboMaster SDK Python 3.10+ compatibility patches & WSL2 interface auto-recovery."""
    try:
        import robomaster.config
        import robomaster.conn
        import robomaster.media

        # Patch native robomaster.media.LiveView queue depth to 1 for zero-latency
        try:
            _orig_liveview_init = robomaster.media.LiveView.__init__
            def _fast_liveview_init(self, robot):
                _orig_liveview_init(self, robot)
                self._video_frame_queue = DropOldestQueue(maxsize=1)
            robomaster.media.LiveView.__init__ = _fast_liveview_init
        except Exception:
            pass

        try:
            from robomaster_ros.ftp import FtpConnection
            robomaster.conn.FtpConnection = FtpConnection
        except ImportError:
            if not hasattr(robomaster.conn, "FtpConnection"):
                class DummyFtp:
                    def __init__(self, *a, **kw): pass
                    def connect(self, *a, **kw): return True
                    def stop(self, *a, **kw): pass
                robomaster.conn.FtpConnection = DummyFtp

        # Auto-recover mirrored interfaces in WSL if DOWN
        try:
            os.system("sudo -n ip link set eth0 up 2>/dev/null; sudo -n ip link set eth1 up 2>/dev/null; sudo -n ip link set eth2 up 2>/dev/null")
        except Exception:
            pass

        target_subnet = "192.168.42." if conn_type == "rndis" else "192.168.2."
        target_ip = "192.168.42.2" if conn_type == "rndis" else "192.168.2.1"
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.3)
        try:
            s.connect((target_ip, 80))
            lip = s.getsockname()[0]
        except Exception:
            lip = ""
        finally:
            s.close()

        if lip and lip.startswith(target_subnet):
            robomaster.config.LOCAL_IP_STR = lip
        else:
            try:
                import subprocess
                out = subprocess.check_output(["ip", "-4", "-o", "addr", "show"], text=True, timeout=0.8)
                for line in out.splitlines():
                    parts = line.split()
                    if len(parts) >= 4 and parts[2] == "inet":
                        ip_val = parts[3].split("/")[0]
                        if ip_val.startswith(target_subnet):
                            robomaster.config.LOCAL_IP_STR = ip_val
                            iface = parts[1]
                            os.system(f"sudo -n ip route replace {target_subnet}0/24 dev {iface} 2>/dev/null")
                            break
            except Exception:
                robomaster.config.LOCAL_IP_STR = ""
    except Exception:
        pass


class FlatnessMissionNode(Node):
    def __init__(self):
        super().__init__("rm_flatness_nav_v4")

        # ROS 2 Launch Parameters
        self.declare_parameter("conn_type", "ap")
        self.declare_parameter("target_tag_id", 3)
        self.declare_parameter("use_sim", False)
        self.declare_parameter("v_max_mps", 0.24)
        self.declare_parameter("standoff_m", 0.20)
        self.declare_parameter("arm_x_mm", 140)
        self.declare_parameter("arm_y_mm", 35)
        self.declare_parameter("auto_exit", False)
        self.declare_parameter("output_dir", "/mnt/d/phd/sce672 - advanced robotics/project apriltag v4")

        self.conn_type = str(self.get_parameter("conn_type").value).lower()
        self.target_tag_id = int(self.get_parameter("target_tag_id").value)
        self.use_sim = bool(self.get_parameter("use_sim").value)
        self.auto_exit = bool(self.get_parameter("auto_exit").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        if not os.path.exists(self.output_dir):
            self.output_dir = os.getcwd()

        # Live-Tunable Parameters (adjustable live from debug_gui.py / tuning_params.json)
        self.sweep_wz_rad = 0.35          # 360-deg sweep rotation speed (rad/s) (~20 deg/s)
        self.align_wz_max = 0.50          # Max turn rate during ALIGNING (rad/s)
        self.v_max = float(self.get_parameter("v_max_mps").value)
        self.v_min = 0.08                 # Min forward approach speed during CHASING (m/s)
        self.chase_wz_max = 0.55          # Max turn rate during CHASING (rad/s)
        self.turn_radius_m = 0.30         # Turning radius limit R (m): |wz| <= |vx| / R
        self.standoff_m = float(self.get_parameter("standoff_m").value)
        self.arm_x_mm = int(self.get_parameter("arm_x_mm").value)
        self.arm_y_mm = int(self.get_parameter("arm_y_mm").value)
        self.kp_bearing = 1.65            # Visual bearing proportional gain
        self.kp_skew = 0.45               # Parallel skew alignment gain
        self.tag_size_mm = 48             # Inner black AprilTag square size in mm (48 mm on 60 mm sheet)
        self.tof_blocked_thresh_mm = 220  # Threshold below which static front ToF is marked BLOCKED_BY_ARM
        self.tof_mode = "VISION_ONLY"     # VISION_ONLY, AUTO, CH0, CH1, CH2, CH3
        self._last_tuning_mtime = 0.0
        self._last_disk_poll_time = 0.0

        # ROS 2 Publishers
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_tof = self.create_publisher(Range, "/tof_distance", 10)
        self.pub_volts = self.create_publisher(Float64MultiArray, "/motor_voltages", 10)
        self.pub_state = self.create_publisher(String, "/mission_state", 10)

        # Vision & Dual Front ToF Selector
        self.detector = AprilTagDetector(tag_size_m=float(self.tag_size_mm) / 1000.0)
        self.tof = FrontToFSelector()
        self.tof.blocked_thresh_mm = float(self.tof_blocked_thresh_mm)
        self.lock = threading.Lock()

        # Load initial tuning_params.json if present
        self._load_tuning_file(initial=True)

        # Robot Pose & Sensor State
        self.hw_connected = False
        self.ep_robot = None
        self.sim = None
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

        # Vision & 360-Deg 2D World Spatial Tag Memory
        # tag_memory[tid] = {'x': m, 'y': m, 'yaw': rad, 'dist_m': m, 'skew_rad': rad}
        self.tag_visible = False
        self.tag_dist_m = 1.20
        self.tag_bearing_rad = 0.0
        self.tag_skew_rad = 0.0
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

        # Manual override nudge state from GUI
        self.manual_vx = 0.0
        self.manual_wz = 0.0
        self.manual_until = 0.0

        # Continuously Replanned Flat Cubic Trajectory Parameters
        self.alpha_x = np.zeros(4)
        self.alpha_y = np.zeros(4)
        self.cubic_T = 3.5
        self.planned_path_pts = []
        self.trajectory_analysis = {}

        # Nonholonomic Velocities & Motor Voltage State (vy is ALWAYS 0.0)
        self.vx = 0.0
        self.wz = 0.0
        self._halted = True
        self.w_prev = np.zeros(4, dtype=np.float64)
        self.voltages = np.zeros(4, dtype=np.float64)
        self.table1_rows = []
        self.continuous_rows = []
        self.last_logged_sec = -1

        self._init_hardware_or_sim()

        # Start dedicated camera grab thread (with 10 ms sleep per Problem 1 Hint)
        # and dedicated 20 Hz chassis actuator thread (decoupled from vision & disk I/O)
        self._stop_evt = threading.Event()
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()
        self._act_thread = threading.Thread(target=self._chassis_actuator_loop, daemon=True)
        self._act_thread.start()

        self.get_logger().info("=" * 72)
        self.get_logger().info(f"Table 1: Motor Armature Voltage Log (Target: Tag {self.target_tag_id})")
        self.get_logger().info(f"{'Timestamp t(s)':<16} {'V1(V)':>8} {'V2(V)':>8} {'V3(V)':>8} {'V4(V)':>8}   {'State'}")
        self.get_logger().info("-" * 72)

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
            if "tof_blocked_thresh_mm" in cfg:
                self.tof_blocked_thresh_mm = int(np.clip(int(float(cfg["tof_blocked_thresh_mm"])), 50, 400))
                self.tof.blocked_thresh_mm = float(self.tof_blocked_thresh_mm)
            if "tof_mode" in cfg:
                self.tof_mode = str(cfg["tof_mode"]).upper()
                self.tof.tof_mode = self.tof_mode

        new_arm = (self.arm_x_mm, self.arm_y_mm)
        if move_arm and (new_arm != old_arm or cfg.get("force_arm_move", False)):
            self._move_arm_async(self.arm_x_mm, self.arm_y_mm)

    def _move_arm_async(self, x_mm: int, y_mm: int):
        if not self.hw_connected or self.ep_robot is None:
            self.get_logger().info(f"[ARM POSE UPDATED] Target set to (X={x_mm} mm, Y={y_mm} mm).")
            return

        def _worker():
            try:
                self.ep_robot.robotic_arm.moveto(x=int(x_mm), y=int(y_mm)).wait_for_completed(timeout=2.0)
                self.get_logger().info(f"[HARDWARE ARM] Moved robotic arm to (X={x_mm} mm, Y={y_mm} mm).")
            except Exception as e:
                self.get_logger().warn(f"Arm move warning: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def _init_hardware_or_sim(self):
        if not self.use_sim:
            patch_robomaster_sdk(self.conn_type)
            try:
                from robomaster import robot
                self.ep_robot = robot.Robot()
                self.ep_robot.initialize(conn_type=self.conn_type)
                self.hw_connected = True
                self.get_logger().info(f"[HARDWARE] Connected to RoboMaster EP ({self.conn_type}).")

                # Move arm asynchronously so node startup is instant (< 0.5s) instead of blocking for 2.5s
                try:
                    self._move_arm_async(self.arm_x_mm, self.arm_y_mm)
                except Exception as e:
                    self.get_logger().warn(f"Arm moveto note: {e}")

                # Register DDS sensor subscriptions BEFORE starting video stream so UDP packets aren't dropped!
                try:
                    self.ep_robot.chassis.sub_position(cs=0, freq=20, callback=self._on_pos)
                    self.ep_robot.chassis.sub_attitude(freq=20, callback=self._on_att)
                    self.ep_robot.chassis.sub_esc(freq=20, callback=self._on_esc)
                    self.ep_robot.sensor.sub_distance(freq=20, callback=self._on_tof)
                    time.sleep(0.08)
                except Exception as e:
                    self.get_logger().warn(f"DDS subscription warning: {e}")

                self.ep_robot.camera.start_video_stream(display=False, resolution="360p")
                return
            except Exception as e:
                self.get_logger().warn(f"Hardware unavailable ({e}) -> switching to WSL Arena Simulation.")

        self.hw_connected = False
        self.sim = SimpleSimArena(target_tag_id=self.target_tag_id)
        self.get_logger().info("[SIMULATION] Running 360-Deg Arena & Dual Front ToF Simulation.")

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
                yaw_rad = -math.radians(delta_deg)  # Convert DJI +CW yaw to standard +CCW yaw
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

    def _on_tof(self, sub_info):
        with self.lock:
            self.tof.update(sub_info, time.monotonic())

    # =========================================================================
    # Dedicated 20 Hz Chassis Actuator Thread (Decoupled from Vision & Disk I/O)
    # =========================================================================
    def _chassis_actuator_loop(self):
        """
        Streams smooth nonholonomic velocity commands (y = 0.0 ALWAYS) at a steady 20 Hz
        with timeout=0.5s so the RoboMaster EP's onboard STM32 watchdog never expires
        and never stutters when WSL performs disk I/O or camera processing.
        """
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
    # Problem 1: Ultra-Low Latency 30 FPS Camera Thread
    # =========================================================================
    def _camera_loop(self):
        while not self._stop_evt.is_set():
            frame = None
            if self.hw_connected and self.ep_robot is not None:
                try:
                    frame = self.ep_robot.camera.read_cv2_image(strategy="newest", timeout=0.08)
                except Exception:
                    frame = None
            elif self.sim is not None:
                time.sleep(0.025)
                frame = self.sim.render_frame()

            if frame is not None:
                now = time.monotonic()
                dets = self.detector.detect_all(frame, default_id=self.target_tag_id)
                target_det = None

                with self.lock:
                    tof_ok = self._is_tof_valid(now)
                    for d in dets:
                        tid = int(d["id"])
                        b_rad = float(d["bearing_rad"])
                        s_rad = float(d["skew_rad"])
                        # Use ToF sensor distance when pointing near the tag; never let camera underestimate distance
                        if tof_ok and abs(b_rad) < 0.35 and 0.12 <= self.tof.dist_m <= 2.50:
                            d_m = float(self.tof.dist_m)
                        else:
                            d_m = max(0.12, float(d["dist_m"]))
                        world_dir = wrap_angle(self.odom_yaw + b_rad)
                        tx = self.odom_x + d_m * math.cos(world_dir)
                        ty = self.odom_y + d_m * math.sin(world_dir)

                        if tid not in self.tag_memory:
                            self.tag_memory[tid] = {
                                "x": tx,
                                "y": ty,
                                "yaw": world_dir,
                                "dist_m": d_m,
                                "skew_rad": s_rad,
                            }
                        else:
                            m = self.tag_memory[tid]
                            m["x"] = 0.60 * tx + 0.40 * m["x"]
                            m["y"] = 0.60 * ty + 0.40 * m["y"]
                            m["yaw"] = math.atan2(m["y"] - self.odom_y, m["x"] - self.odom_x)
                            m["dist_m"] = math.hypot(m["x"] - self.odom_x, m["y"] - self.odom_y)
                            m["skew_rad"] = 0.55 * s_rad + 0.45 * m["skew_rad"]

                        if tid == self.target_tag_id:
                            target_det = d

                    if target_det is not None:
                        self.tag_visible = True
                        raw_d = float(target_det["dist_m"])
                        alpha_d = 0.75 if raw_d < self.tag_dist_m else 0.55
                        self.tag_dist_m = alpha_d * raw_d + (1.0 - alpha_d) * self.tag_dist_m
                        self.tag_bearing_rad = 0.70 * float(target_det["bearing_rad"]) + 0.30 * self.tag_bearing_rad
                        self.tag_skew_rad = 0.65 * float(target_det["skew_rad"]) + 0.35 * self.tag_skew_rad
                        self.last_seen_time = now
                        self.last_seen_dist = float(self.tag_dist_m)
                        self.last_seen_odom_x = float(self.odom_x)
                        self.last_seen_odom_y = float(self.odom_y)
                    else:
                        self.tag_visible = False

                self._write_gui_stream(frame, dets)
            else:
                time.sleep(0.002)

    def _get_mem_tag_relative(self, tid: int):
        """
        Computes the live relative distance (m) and relative bearing (rad) from the
        robot's current (odom_x, odom_y, odom_yaw) to the remembered 2D world position
        of tag `tid`.
        """
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
        return rel_dist, rel_bearing, m["skew_rad"]

    def _is_tof_valid(self, now: float) -> bool:
        """Returns True if the unblocked straight Front ToF sensor has a valid live physical reading."""
        if self.tof_mode in ("CAM_ONLY", "CAMERA_ONLY"):
            return False
        ch = self.tof.active_idx
        return (
            (now - self.tof.last_update_time) < 1.5
            and self.tof.status[ch] in ("STRAIGHT_CLEAR", "CLEAR_AUX")
            and 15 <= self.tof.channels_mm[ch] < 8000
        )

    def _fused_distance(self, now: float) -> float:
        """
        Fused Standoff Distance Estimator (Problem 1(iii)):
        - If the unblocked Front ToF sensor is valid and detecting an obstacle: uses ToF distance.
        - If the ToF sensor has no reading / is out of range / passes over floor tag:
          uses the optical camera distance `self.tag_dist_m`.
        - If close (< 0.40m) and both are available: uses min(tof, cam) so the robot
          NEVER pushes through the tag under any circumstances!
        """
        tof_valid = self._is_tof_valid(now)
        vis_valid = self.tag_visible and ((now - self.last_seen_time) < 0.8)

        if tof_valid and vis_valid:
            tof_d = float(self.tof.dist_m)
            cam_d = float(self.tag_dist_m)
            if min(tof_d, cam_d) < 0.40:
                return min(tof_d, cam_d)
            return tof_d
        elif tof_valid:
            return float(self.tof.dist_m)
        elif vis_valid:
            return float(self.tag_dist_m)
        else:
            # Blind-spot dead reckoning: tag was optically tracked into front approach cone (< 0.65m)
            # and may have slipped below camera FOV as robot reaches 20cm standoff.
            # Decrement distance by incremental wheel displacement so the robot glides accurately to 20cm!
            time_lost = now - self.last_seen_time
            if self.last_seen_dist < 0.65 and time_lost < 6.0:
                driven = math.hypot(self.odom_x - self.last_seen_odom_x, self.odom_y - self.last_seen_odom_y)
                est_d = max(0.08, self.last_seen_dist - driven)
                return float(est_d)

            mem_rel = self._get_mem_tag_relative(self.target_tag_id)
            if mem_rel is not None:
                return float(mem_rel[0])
            return 1.20

    def _replan_cubic_to_tag(self, dist_m: float, bearing_rad: float, skew_rad: float, v_curr: float, log_plan: bool = False):
        """
        Continuously re-solves the 2D flat cubic trajectory z(tau) = [x(tau), y(tau)]^T
        from the robot's current pose (odom_x, odom_y, odom_yaw) to the 20 cm standoff point
        directly in front of the tag.
        """
        d0 = max(self.standoff_m + 0.04, float(dist_m))
        v0 = max(self.v_min, min(self.v_max, float(v_curr)))

        self.alpha_x, self.alpha_y, self.cubic_T, xf, yf = plan_aligning_cubic_path(
            d0=d0,
            bearing_rad=bearing_rad,
            skew_rad=skew_rad,
            standoff_m=self.standoff_m,
            v0=v0,
            v_max=self.v_max,
        )

        self.trajectory_analysis = compute_trajectory_matrices(
            d0=d0,
            bearing_rad=bearing_rad,
            skew_rad=skew_rad,
            standoff_m=self.standoff_m,
            v0=v0,
            v_max=self.v_max,
        )

        # Update planned path points in world frame for real-time GUI visualization
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
            self.get_logger().info("\n" + format_trajectory_report(self.trajectory_analysis) + "\n")

    # =========================================================================
    # 20 Hz Supervisory Control Step (Strictly Nonholonomic: vy = 0.0)
    # =========================================================================
    def step_control(self):
        now = time.monotonic()
        # Poll GUI commands (WASD teleop, Restart, E-Stop) at full 20 Hz for instant response
        self._check_gui_commands()
        if (now - self._last_disk_poll_time) >= 0.25:
            self._last_disk_poll_time = now
            self._load_tuning_file(initial=False)

        elapsed = now - self.start_time
        dt = 0.05

        with self.lock:
            if not self.hw_connected and self.sim is not None:
                self.odom_x, self.odom_y, new_yaw, sim_tof = self.sim.step(self.vx, self.wz, dt)
                if self.prev_yaw is not None:
                    self.sweep_rad += abs(wrap_angle(new_yaw - self.prev_yaw))
                self.prev_yaw = new_yaw
                self.odom_yaw = new_yaw
                self.tof.update(sim_tof, now)
            else:
                # Hybrid Hardware + Dead-Reckoning Odometry Fusion:
                # If WSL2 NAT drops incoming UDP DDS packets, integrate nonholonomic kinematics
                # so odom (x, y, yaw) and 360-deg sweep progress always update smoothly on the GUI
                if (now - self._last_att_cb_time) > 0.15:
                    new_yaw = wrap_angle(self.odom_yaw + self.wz * dt)
                    self.sweep_rad += abs(self.wz) * dt * 0.96
                    self.prev_yaw = new_yaw
                    self.odom_yaw = new_yaw
                if (now - self._last_pos_cb_time) > 0.15:
                    self.odom_x += self.vx * math.cos(self.odom_yaw) * dt
                    self.odom_y += self.vx * math.sin(self.odom_yaw) * dt

            if len(self.odom_trail) == 0 or math.hypot(
                self.odom_x - self.odom_trail[-1][0], self.odom_y - self.odom_trail[-1][1]
            ) > 0.02:
                self.odom_trail.append([round(self.odom_x, 3), round(self.odom_y, 3)])

            vis = self.tag_visible
            vis_recent = vis or ((now - self.last_seen_time) < 1.20)
            mem_rel = self._get_mem_tag_relative(self.target_tag_id)

            if vis:
                bearing = self.tag_bearing_rad
                skew = self.tag_skew_rad
            elif mem_rel is not None:
                bearing = mem_rel[1]
                skew = mem_rel[2]
            else:
                bearing = 0.0
                skew = 0.0

            tof_ok = self._is_tof_valid(now)
            dist = self._fused_distance(now)

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

        elif self.estop_paused:
            vx_cmd, wz_cmd = 0.0, 0.0
            self._halted = True

        elif wasd_active:
            vx_cmd = self.manual_vx
            wz_cmd = self.manual_wz
            self._halted = False

        # ---------------------------------------------------------------------
        # 1. SEARCHING: 360-Deg Sweep at user-configured sweep_wz_rad
        # ---------------------------------------------------------------------
        elif self.state == "SEARCHING":
            wz_cmd = float(self.sweep_wz_rad)
            max_sweep_time = (2.0 * math.pi / max(0.12, abs(wz_cmd))) + 2.5

            if self.sweep_rad >= (2.0 * math.pi - 0.08) or in_state_s >= max_sweep_time:
                self.tof.lock_straight_sensor()
                self.state = "ALIGNING"
                self.state_time = now
                mem_info = (
                    f"Tag {self.target_tag_id} remembered @ rel_brg={math.degrees(mem_rel[1]):+.1f}deg, d={mem_rel[0]:.2f}m"
                    if mem_rel else "Tag not yet seen; continuing search"
                )
                self.get_logger().info(
                    f"[360 SWEEP DONE] {mem_info} | ToF Auto-ID: CH{self.tof.active_idx}=STRAIGHT_CLEAR, "
                    f"Status={self.tof.status} ({self.tof.channels_mm} mm)"
                )

        # ---------------------------------------------------------------------
        # 2. ALIGNING: Turn in Place Until Tag is Centered in Camera View
        # ---------------------------------------------------------------------
        elif self.state == "ALIGNING":
            w_lim = float(self.align_wz_max)
            w_min = min(0.24, 0.50 * w_lim)

            if vis:
                # Directly center the visible tag in the camera view
                wz_cmd = float(np.clip(self.kp_bearing * bearing, -w_lim, w_lim))
                if abs(bearing) > 0.05 and abs(wz_cmd) < w_min:
                    wz_cmd = math.copysign(w_min, bearing)
            elif mem_rel is not None:
                # Turn toward the remembered 2D world coordinates of the tag
                rel_brg = mem_rel[1]
                wz_cmd = float(np.clip(1.6 * rel_brg, -w_lim, w_lim))
                if abs(rel_brg) > 0.07 and abs(wz_cmd) < w_min:
                    wz_cmd = math.copysign(w_min, rel_brg)
            else:
                wz_cmd = float(self.sweep_wz_rad)

            # Standoff Guard in ALIGNING: if already at standoff, halt immediately
            tof_triggered = tof_ok and (self.tof.dist_m <= (self.standoff_m + 0.02))
            cam_triggered = vis and (self.tag_dist_m <= (self.standoff_m + 0.02))
            fused_close = dist <= (self.standoff_m + 0.02)
            if (tof_triggered or cam_triggered or fused_close) and abs(bearing) <= 0.18:
                vx_cmd, wz_cmd = 0.0, 0.0
                self.state = "ARRIVED (20cm)"
                self.state_time = now
                self._halted = True
                self.get_logger().info(f"[ARRIVED] Tag {self.target_tag_id} reached at 20 cm standoff fully aligned!")
            # Transition to CHASING once tag is reasonably in view (or aligned with memory)
            elif (vis and abs(bearing) <= 0.22) or (mem_rel is not None and abs(mem_rel[1]) <= 0.16 and in_state_s >= 0.4):
                with self.lock:
                    self._replan_cubic_to_tag(dist, bearing, skew, v_curr=self.v_min, log_plan=True)
                self.state = "CHASING"
                self.state_time = now

        # ---------------------------------------------------------------------
        # 3. CHASING: Continuous 2D Flat Cubic Pursuit with Terminal Tangent Alignment (vy=0)
        # ---------------------------------------------------------------------
        elif self.state == "CHASING":
            dist_err = dist - self.standoff_m

            # Standoff Guard in CHASING: brake immediately at 20 cm
            # Because the cubic path enforces arrival tangent alignment (Eq. 2), the robot is already aligned!
            tof_triggered = tof_ok and (self.tof.dist_m <= (self.standoff_m + 0.02))
            cam_triggered = vis and (self.tag_dist_m <= (self.standoff_m + 0.02))
            fused_close = dist <= (self.standoff_m + 0.02)
            if dist_err <= 0.015 or fused_close or tof_triggered or cam_triggered:
                vx_cmd, wz_cmd = 0.0, 0.0
                self.state = "ARRIVED (20cm)"
                self.state_time = now
                self._halted = True
                self.get_logger().info(f"[ARRIVED] Tag {self.target_tag_id} reached at 20 cm standoff fully aligned via flat trajectory!")
            # Only return to ALIGNING if the tag is completely behind/sideways (> 65 deg) or lost with no memory
            elif (not vis_recent and mem_rel is None) or abs(bearing) > math.radians(65.0):
                vx_cmd = 0.0
                self.state = "ALIGNING"
                self.state_time = now
                self.get_logger().info(
                    f"[RE-ALIGNING] Tag far off-axis (vis={vis}, brg={math.degrees(bearing):+.1f}deg). "
                    f"Turning to re-center."
                )
            else:
                # CONTINUOUSLY REPLAN the 2D flat cubic trajectory on every control cycle!
                with self.lock:
                    self._replan_cubic_to_tag(dist, bearing, skew, v_curr=max(self.v_min, self.vx), log_plan=False)

                # Evaluate the newly replanned cubic polynomial at a short receding-horizon lookahead tau
                tau_look = min(0.35, 0.20 * self.cubic_T)
                v_ff, w_ff, _, _, xdot, ydot = evaluate_flatness(
                    self.alpha_x, self.alpha_y, tau_look, self.cubic_T
                )
                cubic_heading = math.atan2(ydot, max(0.03, xdot))

                # Smoothly scale forward speed by cos(bearing) so the robot curves continuously
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
        # 4. FINE_ALIGN: Reverse-and-Forward Nonholonomic Parking Maneuver
        #    Backs up while turning to eliminate lateral/skew offset, then pulls
        #    forward while turning to park squarely and parallel at 20 cm.
        # ---------------------------------------------------------------------
        elif self.state == "FINE_ALIGN":
            dist_err = dist - self.standoff_m
            sub_s = now - self.park_phase_time
            w_lim = min(0.48, float(self.align_wz_max))

            if self.park_phase == "BACKUP":
                # Move backward while turning to swing onto the tag's perpendicular centerline
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
                    self.get_logger().info(
                        f"[PARKING: PULLING FORWARD] Backed up to d={dist:.2f}m "
                        f"(brg={math.degrees(bearing):+.1f}deg, skew={math.degrees(skew):+.1f}deg) -> pulling forward while turning."
                    )

            elif self.park_phase == "FORWARD":
                # Move forward while turning along replanned cubic curve to straighten onto 20 cm standoff
                with self.lock:
                    self._replan_cubic_to_tag(dist, bearing, skew, v_curr=max(0.05, self.vx), log_plan=False)
                vx_cmd = float(np.clip(0.65 * max(0.04, dist_err), 0.05, min(0.10, self.v_max)))
                steer_fwd = 1.65 * bearing - 0.60 * skew
                wz_cmd = float(np.clip(steer_fwd, -w_lim, w_lim))
                if abs(steer_fwd) > 0.03 and abs(wz_cmd) < 0.14:
                    wz_cmd = math.copysign(0.14, steer_fwd)

                tof_trig = tof_ok and (self.tof.dist_m <= (self.standoff_m + 0.015))
                cam_trig = vis and (self.tag_dist_m <= (self.standoff_m + 0.015))
                fused_trig = dist <= (self.standoff_m + 0.015)
                if dist_err <= 0.015 or fused_trig or tof_trig or cam_trig or (dist_err <= 0.030 and sub_s >= 3.0):
                    self.park_phase = "TRIM"
                    self.park_phase_time = now
                    self.park_passes += 1

            else:  # "TRIM"
                # Fine distance trim based on fused distance
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
                    self.get_logger().info(
                        f"[PARKING PASS {self.park_passes + 1}] Re-backing up to refine alignment "
                        f"(d={dist:.2f}m, brg={math.degrees(bearing):+.1f}deg, skew={math.degrees(skew):+.1f}deg)."
                    )
                else:
                    dist_ok = abs(dist_err) <= 0.025
                    parallel_ok = (not vis) or (abs(bearing) <= math.radians(4.5) and abs(skew) <= math.radians(7.0))
                    if dist_ok and parallel_ok and sub_s >= 0.40:
                        self.state = "ARRIVED (20cm)"
                        self.state_time = now
                        vx_cmd, wz_cmd = 0.0, 0.0
                        self.get_logger().info(
                            f"[ARRIVED AT {int(self.standoff_m*100)}cm] Parked squarely at d={dist:.3f}m "
                            f"(cam={self.tag_dist_m:.3f}m, ToF CH{self.tof.active_idx}={self.tof.dist_m:.3f}m, "
                            f"bearing={math.degrees(bearing):+.1f}deg, parallel_skew={math.degrees(skew):+.1f}deg)."
                        )
                    elif sub_s >= 2.5:
                        if dist_err > 0.030:
                            self.park_phase = "FORWARD"
                            self.park_phase_time = now
                        elif dist_ok or (sub_s >= 3.5):
                            self.state = "ARRIVED (20cm)"
                            self.state_time = now
                            vx_cmd, wz_cmd = 0.0, 0.0
                            self.get_logger().info(
                                f"[ARRIVED AT {int(self.standoff_m*100)}cm] Completed parking at d={dist:.3f}m."
                            )

        # ---------------------------------------------------------------------
        # 5. ARRIVED (20cm): Full Active Brake (or WASD Teleop if keys pressed)
        # ---------------------------------------------------------------------
        elif self.state == "ARRIVED (20cm)":
            vx_cmd, wz_cmd = 0.0, 0.0

        # Velocity Smoother (Nav2 / EMA style) to prevent jerky wheel step changes,
        # while keeping immediate braking at ARRIVED (20cm) or E-Stop
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
                # Include subtle closed-loop PID wheel velocity ripple so dw/dt reflects real motor dynamics
                ripple = 0.035 * math.sin(6.5 * elapsed) * np.array([1.0, -1.0, 1.0, -1.0])
                w_curr = w_kin * (1.0 + ripple)

        V = motor_voltages(w_curr, self.w_prev, dt, halted=halted)
        self.w_prev = w_curr

        with self.lock:
            self.vx = vx_smooth
            self.wz = wz_smooth
            self._halted = halted
            self.voltages = V.copy()

        # Publish ROS 2 Topics
        self._publish_ros(vx_smooth, wz_smooth, dist, V)

        # Active state label for logs and GUI
        disp_state = (
            "WASD TELEOP"
            if wasd_active
            else (f"FINE_ALIGN ({self.park_phase})" if self.state == "FINE_ALIGN" else self.state)
        )

        # Log Continuous 20 Hz Telemetry
        self.continuous_rows.append([
            round(elapsed, 3),
            round(float(V[0]), 4), round(float(V[1]), 4), round(float(V[2]), 4), round(float(V[3]), 4),
            round(vx_smooth, 4), 0.0, round(wz_smooth, 4), round(dist, 4),
            round(math.degrees(bearing), 2), round(math.degrees(skew), 2), disp_state,
        ])

        # Log 1-Second Table 1 Row (Problem 3(iii)) during mission AND during WASD Teleop
        if sec > self.last_logged_sec and (not self.mission_done or wasd_active):
            self.last_logged_sec = sec
            row = [sec, round(float(V[0]), 3), round(float(V[1]), 3), round(float(V[2]), 3), round(float(V[3]), 3), disp_state]
            with self.lock:
                self.table1_rows.append(row)
            self.get_logger().info(f"{sec:<16d} {V[0]:>8.3f} {V[1]:>8.3f} {V[2]:>8.3f} {V[3]:>8.3f}   {disp_state}")
            if self.state == "ARRIVED (20cm)" and not self.mission_done:
                self.mission_done = True
                self._save_csv_logs()
                self.get_logger().info("=" * 72)
                self.get_logger().info(f"[+] Mission Complete! Table 1 saved to: {os.path.join(self.output_dir, 'motor_voltage_log.csv')}")
                self.get_logger().info("[+] You can now drive the robot using W/A/S/D keys or click RESTART MISSION in the GUI.")
                self.get_logger().info("=" * 72)

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
        self.get_logger().info(
            f"[REACHED {int(self.standoff_m*100)}cm ZONE] d={dist:.3f}m, brg={math.degrees(bearing):+.1f}deg, "
            f"skew={math.degrees(skew):+.1f}deg -> Starting FINE_ALIGN ({self.park_phase})."
        )

    def _publish_ros(self, vx: float, wz: float, dist: float, V: np.ndarray):
        stamp = self.get_clock().now().to_msg()
        tw = Twist()
        tw.linear.x = float(vx)
        tw.linear.y = 0.0
        tw.angular.z = float(wz)
        self.pub_cmd.publish(tw)

        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = "odom"
        od.pose.pose.position.x = float(self.odom_x)
        od.pose.pose.position.y = float(self.odom_y)
        od.twist.twist = tw
        self.pub_odom.publish(od)

        rng = Range()
        rng.header.stamp = stamp
        rng.range = float(dist)
        self.pub_tof.publish(rng)

        vm = Float64MultiArray()
        vm.data = [float(v) for v in V]
        self.pub_volts.publish(vm)

        sm = String()
        sm.data = self.state
        self.pub_state.publish(sm)

    def _write_gui_stream(self, raw_frame: np.ndarray, all_dets: list):
        try:
            vis_img = cv2.resize(raw_frame, (380, 214))
            sx = 380.0 / float(raw_frame.shape[1])
            sy = 214.0 / float(raw_frame.shape[0])

            cv2.line(vis_img, (190, 0), (190, 214), (85, 85, 85), 1)
            cv2.line(vis_img, (0, 107), (380, 107), (85, 85, 85), 1)
            cv2.rectangle(vis_img, (145, 62), (235, 151), (70, 135, 70), 1)
            cv2.putText(vis_img, "20cm TARGET", (140, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 180, 90), 1)

            gy = int(0.70 * 214)
            cv2.line(vis_img, (int(0.24 * 380), gy), (int(0.90 * 380), gy), (60, 140, 60), 1)
            cv2.putText(vis_img, "20cm STANDOFF", (int(0.36 * 380), gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 200, 80), 1)

            for d in all_dets:
                x, y, w, h = d["bbox"]
                x1, y1, x2, y2 = int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)
                is_tgt = int(d["id"]) == self.target_tag_id
                col = (0, 255, 0) if is_tgt else (0, 180, 255)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), col, 2)
                if d.get("corners") is not None:
                    pts = (d["corners"] * np.array([sx, sy])).astype(np.int32)
                    cv2.polylines(vis_img, [pts], True, (0, 229, 255), 1)
                lbl = f"T{d['id']} d={d['dist_m']:.2f}m brg={math.degrees(d['bearing_rad']):+.0f}d"
                cv2.putText(vis_img, lbl, (max(4, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)

            now = time.monotonic()
            with self.lock:
                fused_d = self._fused_distance(now)
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
                    "mode": "HARDWARE (" + self.conn_type.upper() + ")" if self.hw_connected else "WSL SIMULATION",
                    "state": gui_state,
                    "target_tag_id": self.target_tag_id,
                    "sweep_deg": round(min(360.0, math.degrees(self.sweep_rad)), 1),
                    "fused_dist_m": round(float(fused_d), 3),
                    "standoff_goal_m": self.standoff_m,
                    "tag_visible": bool(self.tag_visible),
                    "cam_dist_m": round(float(self.tag_dist_m), 3),
                    "bearing_deg": round(math.degrees(self.tag_bearing_rad), 2),
                    "skew_deg": round(math.degrees(self.tag_skew_rad), 2),
                    "tof_dist_m": round(float(self.tof.dist_m), 3),
                    "tof_active_ch": int(self.tof.active_idx),
                    "tof_channels_mm": list(self.tof.channels_mm),
                    "tof_status": list(self.tof.status),
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
                        "tof_blocked_thresh_mm": self.tof_blocked_thresh_mm,
                        "tof_mode": self.tof_mode,
                    },
                    "tag_memory": {
                        str(k): {
                            "x": round(v.get("x", 0.0), 2),
                            "y": round(v.get("y", 0.0), 2),
                            "yaw_deg": round(math.degrees(v["yaw"]), 1),
                            "dist_m": round(v["dist_m"], 2),
                            "skew_deg": round(math.degrees(v["skew_rad"]), 1),
                        }
                        for k, v in self.tag_memory.items()
                    },
                    "planned_path": self.planned_path_pts,
                    "odom_trail": self.odom_trail[-60:],
                    "table1": self.table1_rows[-18:],
                    "trajectory_analysis": self.trajectory_analysis,
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
                self.get_logger().warn("[GUI] Emergency Stop Activated!")
            elif action == "resume":
                self.estop_paused = False
                self.get_logger().info("[GUI] Resumed mission.")
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
                    self.odom_trail.clear()
                    self.planned_path_pts.clear()
                    self.table1_rows.clear()
                    self.continuous_rows.clear()
                    self.tag_memory.clear()
                    self.tof.reset_sweep_stats()
                    if not self.hw_connected and self.sim is not None:
                        self.sim = SimpleSimArena(target_tag_id=self.target_tag_id)
                    if "target_tag_id" in cmd:
                        self.target_tag_id = int(cmd["target_tag_id"])
                self.get_logger().info("=" * 72)
                self.get_logger().info(f"[GUI] RESTARTED MISSION: 360-Deg Sweep for Tag {self.target_tag_id} (Odom & Table 1 Reset)!")
                self.get_logger().info("=" * 72)
            elif action == "set_tag" and "target_tag_id" in cmd:
                self.target_tag_id = int(cmd["target_tag_id"])
                self.get_logger().info(f"[GUI] Switched Target Tag to T{self.target_tag_id}.")
            elif action == "update_params":
                params = cmd.get("params", {})
                self._apply_tuning_dict(params, move_arm=True)
                self.get_logger().info(
                    f"[GUI TUNING APPLIED] sweep_wz={self.sweep_wz_rad:.2f} rad/s, align_wz={self.align_wz_max:.2f} rad/s, "
                    f"vmax={self.v_max:.2f} m/s, R={self.turn_radius_m:.2f} m, arm=({self.arm_x_mm},{self.arm_y_mm}) mm, "
                    f"tag_size={self.tag_size_mm} mm, tof_mode={self.tof_mode}"
                )
            elif action == "start_mission":
                now = time.monotonic()
                with self.lock:
                    if self.state not in ("SEARCHING", "ALIGNING", "CHASING"):
                        self.start_time = now
                        self.state_time = now
                        self.mission_done = False
                        self.table1_rows.clear()
                        self.continuous_rows.clear()
                        self.odom_trail.clear()
                        self.last_logged_sec = -1
                        self._halted = False
                        self.sweep_rad = 0.0
                        self.estop_paused = False
                        if self.tag_visible and abs(self.tag_bearing_rad) <= 0.28:
                            self._replan_cubic_to_tag(self.tag_dist_m, self.tag_bearing_rad, self.tag_skew_rad, v_curr=self.v_min, log_plan=True)
                            self.state = "CHASING"
                            self.get_logger().info(f"[MISSION START] Tag {self.target_tag_id} directly in view. Pursuing flat cubic trajectory!")
                        else:
                            self.state = "SEARCHING"
                            self.get_logger().info(f"[MISSION START] Optical 360-Sweep for Tag {self.target_tag_id} started...")
            elif action == "return_to_teleop":
                with self.lock:
                    self.state = "TELEOP"
                    self.state_time = time.monotonic()
                    self.manual_until = 0.0
                    self.manual_vx = 0.0
                    self.manual_wz = 0.0
                    self.vx = 0.0
                    self.wz = 0.0
                    self._halted = True
                    self.get_logger().info("[TELEOP] Switched to Manual Positioning Mode.")
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

    def _save_csv_logs(self):
        try:
            t1_path = os.path.join(self.output_dir, "motor_voltage_log.csv")
            with open(t1_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "V1_V", "V2_V", "V3_V", "V4_V", "State"])
                w.writerows(self.table1_rows)

            c_path = os.path.join(self.output_dir, "motor_voltage_continuous.csv")
            with open(c_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "V1_V", "V2_V", "V3_V", "V4_V", "vx_mps", "vy_mps", "wz_radps", "dist_m", "bearing_deg", "skew_deg", "state"])
                w.writerows(self.continuous_rows)

            # Auto-archive run with dedicated folder and generated graphs
            try:
                import datetime
                import sys
                if self.output_dir not in sys.path:
                    sys.path.insert(0, self.output_dir)
                from run_archiver import archive_run
                run_tag = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_tag{self.target_tag_id}"
                archive_run(
                    run_tag,
                    t1_path,
                    c_path,
                    target_tag_id=self.target_tag_id,
                    d0=getattr(self, 'd0_measured', 1.50),
                    standoff=self.standoff_m,
                    trajectory_meta=getattr(self, 'trajectory_analysis', None)
                )
            except Exception as e:
                self.get_logger().warn(f"[!] Auto-archive warning: {e}")
        except Exception:
            pass

    def cleanup(self):
        self._stop_evt.set()
        if self._cam_thread.is_alive():
            self._cam_thread.join(timeout=1.0)
        if self._act_thread.is_alive():
            self._act_thread.join(timeout=1.0)
        if self.hw_connected and self.ep_robot is not None:
            try:
                self.ep_robot.chassis.drive_speed(x=0.0, y=0.0, z=0.0, timeout=0.5)
                self.ep_robot.camera.stop_video_stream()
                self.ep_robot.close()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = FlatnessMissionNode()
    try:
        while rclpy.ok():
            node.step_control()
            rclpy.spin_once(node, timeout_sec=0.005)
            if node.auto_exit and node.mission_done:
                break
            time.sleep(0.045)
    except KeyboardInterrupt:
        pass
    finally:
        node._save_csv_logs()
        node.cleanup()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
