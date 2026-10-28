#!/usr/bin/env python3
"""
debug_gui.py - Real-Time Native Tkinter + Web Debugger & Live Tuning GUI (v4)
=============================================================================
Non-overflowing, high-density responsive layout for DJI RoboMaster EP (Assessment 2).
Provides live, code-free tuning controls:
  - 360° Sweep Rotation Speed (rad/s & deg/s)
  - Align Rotation Speed & Chase Turning Speed (rad/s)
  - Max & Min Chassis Forward Speed (m/s)
  - Min Turning Radius R (m) during flat cubic pursuit
  - Robotic Arm Pose (Arm X mm, Arm Y mm) with instant "Move Arm" & presets
  - Target Stop Standoff Distance (m, default 0.20 m = 20 cm)
  - Visual Bearing & Parallel Skew Gains (kp_bearing, kp_skew)
  - Front ToF Sensor Mode (AUTO vs. manual CH0..CH3 override + Blocked Threshold mm)
  - WASD Keyboard Teleop & Restart Mission controls (strictly no strafe: vy = 0.0)
  - Motor voltages V1..V4, wheel speeds w1..w4, ToF sensors, and Table 1 log.
"""

import os
import sys
import io
import json
import math
import time
import threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image, ImageTk, ImageDraw

from flatness_math import format_trajectory_report, compute_trajectory_matrices

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) if os.path.basename(SCRIPT_DIR) == "src" else SCRIPT_DIR
BASE_DIR = PROJECT_DIR

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

IMG_PATH = os.path.join(BASE_DIR, "live_camera.jpg")
JSON_PATH = os.path.join(BASE_DIR, "live_telemetry.json")
CMD_PATH = os.path.join(BASE_DIR, "gui_command.json")
TUNING_PATH = os.path.join(BASE_DIR, "config", "tuning_params.json") if os.path.exists(os.path.join(BASE_DIR, "config", "tuning_params.json")) else os.path.join(BASE_DIR, "tuning_params.json")

DEFAULT_TUNING = {
    "target_tag_id": 3,
    "sweep_wz_rad": 0.45,
    "align_wz_max": 0.60,
    "v_max_mps": 0.28,
    "v_min_mps": 0.10,
    "chase_wz_max": 0.55,
    "turn_radius_m": 0.35,
    "standoff_m": 0.20,
    "arm_x_mm": 140,
    "arm_y_mm": 35,
    "kp_bearing": 1.65,
    "kp_skew": 0.45,
    "tag_size_mm": 48,
    "tof_blocked_thresh_mm": 220,
    "tof_mode": "CAM_ONLY",
}


def load_tuning() -> dict:
    cfg = dict(DEFAULT_TUNING)
    if os.path.exists(TUNING_PATH):
        try:
            with open(TUNING_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save_tuning_and_notify(cfg: dict, force_arm_move: bool = False):
    try:
        tmp = TUNING_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, TUNING_PATH)
    except Exception:
        pass
    cmd_payload = dict(cfg)
    if force_arm_move:
        cmd_payload["force_arm_move"] = True
    send_gui_command("update_params", params=cmd_payload)


def send_gui_command(action: str, target_tag_id: int = None, params: dict = None, **kwargs):
    payload = {"action": action, "time": time.time()}
    if target_tag_id is not None:
        payload["target_tag_id"] = int(target_tag_id)
    if params is not None:
        payload["params"] = params
    payload.update(kwargs)
    try:
        tmp = CMD_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, CMD_PATH)
    except Exception:
        pass


class DebugApp:
    def __init__(self, root: tk.Tk, runner=None):
        self.root = root
        self.runner = runner
        self.root.title("DJI RoboMaster EP — v4 Real-Time Camera, Telemetry & Live Tuning GUI")
        self.root.geometry("1280x720")
        self.root.minsize(1050, 580)
        self.root.configure(bg="#0b0f17")

        # Open maximized on Windows for optimal real estate, without taskbar overflow
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        self.tuning = load_tuning()
        self.slider_vars = {}
        self.slider_val_labels = {}
        self.tof_mode_var = tk.StringVar(value=str(self.tuning.get("tof_mode", "CAM_ONLY")))
        self.pressed_keys = set()
        self._was_wasd_moving = False
        self.wasd_btns = {}
        self._last_trajectory_analysis = {}
        self.current_trail = []
        self.history_trails = []

        self.current_state = "TELEOP" if self.runner else "WAITING"
        self.show_cv2_window = False
        self._frame_count = 0
        self._fps_time = time.monotonic()
        self._fps_val = 0.0

        self._build_ui()
        self._bind_keyboard()
        self._tk_cam_img = None
        self._update_loop()

    def _build_ui(self):
        # =====================================================================
        # TOP HEADER: State, Target Selection, Start Mission, Return to Teleop, E-Stop
        # =====================================================================
        top = tk.Frame(self.root, bg="#151c28", highlightbackground="#283347", highlightthickness=1, padx=8, pady=4)
        top.pack(fill=tk.X, padx=8, pady=(6, 3))

        title_box = tk.Frame(top, bg="#151c28")
        title_box.pack(side=tk.LEFT)
        tk.Label(
            title_box,
            text="DJI RoboMaster EP — Assessment 2 (v4) Debugger & Live Tuner",
            fg="#00e5ff", bg="#151c28", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="Real-Time 360° Sweep, Flat Cubic Trajectory, Standoff & Motor Armature Voltages",
            fg="#94a3b8", bg="#151c28", font=("Segoe UI", 8)
        ).pack(anchor="w")

        btn_box = tk.Frame(top, bg="#151c28")
        btn_box.pack(side=tk.RIGHT)

        self.lbl_state = tk.Label(
            btn_box, text="STATE: 🟢 TELEOP / POSITIONING", fg="#00e676", bg="#0f1520",
            font=("Consolas", 10, "bold"), padx=8, pady=3,
            highlightbackground="#283347", highlightthickness=1
        )
        self.lbl_state.pack(side=tk.LEFT, padx=(0, 6))

        # Big Prominent Mission Action Button (Starts Mission, Stops Mission, or Returns to Teleop)
        self.btn_mission_action = tk.Button(
            btn_box, text="🚀 START AUTONOMOUS MISSION [Space]", bg="#059669", fg="#ffffff",
            activebackground="#10b981", activeforeground="#ffffff",
            font=("Segoe UI", 9, "bold"), relief=tk.FLAT, padx=10, pady=2,
            command=self._on_action_button_click
        )
        self.btn_mission_action.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_teleop_return = tk.Button(
            btn_box, text="🎮 TELEOP", bg="#1e293b", fg="#38bdf8",
            activebackground="#334155", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=6, pady=2,
            command=self._return_to_teleop
        )
        self.btn_teleop_return.pack(side=tk.LEFT, padx=(0, 6))

        for tid in (1, 2, 3):
            tk.Button(
                btn_box, text=f"Target T{tid}", bg="#1e293b", fg="#e2e8f0",
                activebackground="#334155", activeforeground="#ffffff",
                font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=5, pady=2,
                command=lambda t=tid: self._set_target_tag(t)
            ).pack(side=tk.LEFT, padx=1)

        tk.Button(
            btn_box, text="↻ RESTART [R]", bg="#0284c7", fg="#ffffff",
            activebackground="#0369a1", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=6, pady=2,
            command=self._restart_mission
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            btn_box, text="■ E-STOP", bg="#dc2626", fg="#ffffff",
            activebackground="#b91c1c", font=("Segoe UI", 8, "bold"), relief=tk.FLAT, padx=6, pady=2,
            command=lambda: send_gui_command("estop")
        ).pack(side=tk.LEFT, padx=(2, 0))

        # =====================================================================
        # MAIN 3-COLUMN RESPONSIVE BODY
        # =====================================================================
        body = tk.Frame(self.root, bg="#0b0f17")
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=3)

        # ---------------------------------------------------------------------
        # COLUMN 1 (Left, ~390px): Camera Feed + 360 Radar & Cubic Map
        # ---------------------------------------------------------------------
        col1 = tk.Frame(body, bg="#0b0f17", width=390)
        col1.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        col1.pack_propagate(False)

        cam_card = tk.Frame(col1, bg="#151c28", highlightbackground="#283347", highlightthickness=1, padx=4, pady=4)
        cam_card.pack(fill=tk.X, pady=(0, 4))
        tk.Label(
            cam_card, text="LIVE ROBOT CAMERA FEED (TAG + PARALLEL SKEW)",
            fg="#94a3b8", bg="#151c28", font=("Segoe UI", 7, "bold")
        ).pack(anchor="w", pady=(0, 2))
        self.cam_label = tk.Label(cam_card, bg="#000000", width=380, height=214)
        self.cam_label.pack()

        cam_ctl = tk.Frame(cam_card, bg="#151c28")
        cam_ctl.pack(fill=tk.X, pady=(2, 0))
        self.btn_popout_cv2 = tk.Button(
            cam_ctl, text="🗖 Pop-out Native Video (OpenCV)",
            bg="#1e293b", fg="#38bdf8", activebackground="#334155",
            font=("Segoe UI", 7, "bold"), relief=tk.FLAT, padx=4, pady=1,
            command=self._toggle_cv2_window
        )
        self.btn_popout_cv2.pack(side=tk.LEFT)
        self.lbl_cam_fps = tk.Label(
            cam_ctl, text="Native Direct Feed | 0ms Latency",
            fg="#00e676", bg="#151c28", font=("Consolas", 7)
        )
        self.lbl_cam_fps.pack(side=tk.RIGHT)

        map_card = tk.Frame(col1, bg="#151c28", highlightbackground="#283347", highlightthickness=1, padx=4, pady=4)
        map_card.pack(fill=tk.BOTH, expand=True)
        map_hdr = tk.Frame(map_card, bg="#151c28")
        map_hdr.pack(fill=tk.X, pady=(0, 2))
        tk.Label(
            map_hdr, text="360° RADAR & 2D FLAT CUBIC MAP",
            fg="#94a3b8", bg="#151c28", font=("Segoe UI", 7, "bold")
        ).pack(side=tk.LEFT)
        self.btn_save_path = tk.Button(
            map_hdr, text="💾 Save Path Plot", bg="#0284c7", fg="#ffffff",
            activebackground="#0369a1", font=("Segoe UI", 7, "bold"), relief=tk.FLAT, padx=6, pady=0,
            command=self._save_path_plot
        )
        self.btn_save_path.pack(side=tk.RIGHT)
        self.map_canvas = tk.Canvas(map_card, width=380, height=180, bg="#0f1520", highlightthickness=0)
        self.map_canvas.pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------------------------------
        # COLUMN 2 (Center, ~420px): Live Sliders (2-Col Grid) & WASD Pad
        # ---------------------------------------------------------------------
        col2 = tk.Frame(body, bg="#151c28", highlightbackground="#00e5ff", highlightthickness=1, padx=8, pady=6, width=420)
        col2.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=6)
        col2.pack_propagate(False)

        hdr_tune = tk.Frame(col2, bg="#151c28")
        hdr_tune.pack(fill=tk.X, pady=(0, 3))
        tk.Label(
            hdr_tune, text="⚙ LIVE TUNING CONTROLS",
            fg="#00e5ff", bg="#151c28", font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT)
        self.lbl_sync = tk.Label(
            hdr_tune, text="● Live Sync", fg="#00e676", bg="#151c28", font=("Consolas", 7, "bold")
        )
        self.lbl_sync.pack(side=tk.RIGHT)

        # 2-Column Grid of 13 Sliders: Motion/Chassis on Left, Arm/Vision on Right
        sliders_grid = tk.Frame(col2, bg="#151c28")
        sliders_grid.pack(fill=tk.X)
        sliders_grid.columnconfigure(0, weight=1)
        sliders_grid.columnconfigure(1, weight=1)

        slider_specs_left = [
            ("sweep_wz_rad", "360° Sweep Wz", 0.15, 1.40, 0.05, lambda v: f"{v:.2f} rad/s ({math.degrees(v):.0f}°/s)"),
            ("align_wz_max", "Max Align Wz", 0.15, 1.50, 0.05, lambda v: f"{v:.2f} rad/s ({math.degrees(v):.0f}°/s)"),
            ("v_max_mps", "Max Fwd Speed", 0.08, 0.55, 0.02, lambda v: f"{v:.2f} m/s"),
            ("v_min_mps", "Min Approach Spd", 0.04, 0.25, 0.01, lambda v: f"{v:.2f} m/s"),
            ("turn_radius_m", "Min Turn Rad R", 0.10, 1.50, 0.05, lambda v: f"{v:.2f} m"),
            ("chase_wz_max", "Max Chase Wz", 0.15, 1.40, 0.05, lambda v: f"{v:.2f} rad/s ({math.degrees(v):.0f}°/s)"),
            ("standoff_m", "Target Standoff", 0.10, 0.50, 0.01, lambda v: f"{v:.2f} m ({int(round(v*100))}cm)"),
        ]

        slider_specs_right = [
            ("arm_x_mm", "Arm Forward X", 60.0, 220.0, 5.0, lambda v: f"{int(round(v))} mm"),
            ("arm_y_mm", "Arm Height Y", -60.0, 160.0, 5.0, lambda v: f"{int(round(v))} mm"),
            ("kp_bearing", "Bearing Gain Kp", 0.40, 3.50, 0.05, lambda v: f"{v:.2f}"),
            ("kp_skew", "Skew Align Gain", 0.00, 1.50, 0.05, lambda v: f"{v:.2f}"),
            ("tag_size_mm", "AprilTag Size", 25.0, 120.0, 1.0, lambda v: f"{int(round(v))} mm"),
            ("tof_blocked_thresh_mm", "Arm ToF Thresh", 60.0, 320.0, 5.0, lambda v: f"< {int(round(v))} mm"),
        ]

        def _make_slider_cell(parent, r, c, spec):
            key, label_txt, vmin, vmax, res, fmt_fn = spec
            cell = tk.Frame(parent, bg="#0f1520", highlightbackground="#283347", highlightthickness=1, padx=4, pady=1)
            cell.grid(row=r, column=c, sticky="nsew", padx=2, pady=1)

            top_r = tk.Frame(cell, bg="#0f1520")
            top_r.pack(fill=tk.X)
            tk.Label(top_r, text=label_txt, fg="#e2e8f0", bg="#0f1520", font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT)
            init_val = float(self.tuning.get(key, DEFAULT_TUNING[key]))
            val_lbl = tk.Label(top_r, text=fmt_fn(init_val), fg="#00e5ff", bg="#0f1520", font=("Consolas", 7, "bold"))
            val_lbl.pack(side=tk.RIGHT)

            var = tk.DoubleVar(value=init_val)
            self.slider_vars[key] = (var, fmt_fn, res >= 1.0)
            self.slider_val_labels[key] = val_lbl

            scale = tk.Scale(
                cell, variable=var, from_=vmin, to=vmax, resolution=res,
                orient=tk.HORIZONTAL, showvalue=False, bg="#0f1520", fg="#00e5ff",
                troughcolor="#1e293b", highlightthickness=0, bd=0, length=180, sliderlength=12, width=8,
                command=lambda _v, k=key: self._on_slider_drag(k)
            )
            scale.bind("<ButtonRelease-1>", lambda _e: self._push_tuning(force_arm=False))
            scale.pack(fill=tk.X)

        for row_i, spec in enumerate(slider_specs_left):
            _make_slider_cell(sliders_grid, row_i, 0, spec)
        for row_i, spec in enumerate(slider_specs_right):
            _make_slider_cell(sliders_grid, row_i, 1, spec)

        # Presets Row: Speed Presets & Arm Presets
        presets_box = tk.Frame(col2, bg="#151c28", pady=2)
        presets_box.pack(fill=tk.X, pady=(3, 1))

        tk.Label(presets_box, text="Presets:", fg="#94a3b8", bg="#151c28", font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT, padx=(0, 2))
        tk.Button(
            presets_box, text="🐢 Slow", bg="#1e293b", fg="#38bdf8", relief=tk.FLAT, font=("Segoe UI", 7, "bold"), padx=4, pady=1,
            command=lambda: self._apply_preset({"sweep_wz_rad": 0.30, "align_wz_max": 0.42, "v_max_mps": 0.18, "chase_wz_max": 0.40, "turn_radius_m": 0.50})
        ).pack(side=tk.LEFT, padx=1)
        tk.Button(
            presets_box, text="⚖ Bal", bg="#1e293b", fg="#00e676", relief=tk.FLAT, font=("Segoe UI", 7, "bold"), padx=4, pady=1,
            command=lambda: self._apply_preset({"sweep_wz_rad": 0.45, "align_wz_max": 0.60, "v_max_mps": 0.28, "chase_wz_max": 0.55, "turn_radius_m": 0.35})
        ).pack(side=tk.LEFT, padx=1)
        tk.Button(
            presets_box, text="⚡ Fast", bg="#1e293b", fg="#ffab00", relief=tk.FLAT, font=("Segoe UI", 7, "bold"), padx=4, pady=1,
            command=lambda: self._apply_preset({"sweep_wz_rad": 0.70, "align_wz_max": 0.90, "v_max_mps": 0.40, "chase_wz_max": 0.80, "turn_radius_m": 0.25})
        ).pack(side=tk.LEFT, padx=1)

        tk.Label(presets_box, text="| Arm:", fg="#64748b", bg="#151c28", font=("Segoe UI", 7)).pack(side=tk.LEFT, padx=2)
        tk.Button(
            presets_box, text="Floor", bg="#1e293b", fg="#e2e8f0", relief=tk.FLAT, font=("Segoe UI", 7), padx=4, pady=1,
            command=lambda: self._set_arm_pose(105, 30)
        ).pack(side=tk.LEFT, padx=1)
        tk.Button(
            presets_box, text="Mid", bg="#1e293b", fg="#e2e8f0", relief=tk.FLAT, font=("Segoe UI", 7), padx=4, pady=1,
            command=lambda: self._set_arm_pose(105, 60)
        ).pack(side=tk.LEFT, padx=1)
        tk.Button(
            presets_box, text="Low", bg="#1e293b", fg="#e2e8f0", relief=tk.FLAT, font=("Segoe UI", 7), padx=4, pady=1,
            command=lambda: self._set_arm_pose(120, 15)
        ).pack(side=tk.LEFT, padx=1)
        tk.Button(
            presets_box, text="🦾 Move Arm", bg="#0284c7", fg="#ffffff", relief=tk.FLAT, font=("Segoe UI", 7, "bold"), padx=4, pady=1,
            command=lambda: self._push_tuning(force_arm=True)
        ).pack(side=tk.RIGHT)

        # Front ToF Mode Selector (CAM_ONLY vs AUTO vs CH0..CH3)
        tof_sel = tk.Frame(col2, bg="#151c28", pady=1)
        tof_sel.pack(fill=tk.X, pady=(1, 2))
        tk.Label(tof_sel, text="Sensor:", fg="#94a3b8", bg="#151c28", font=("Segoe UI", 7, "bold")).pack(side=tk.LEFT, padx=(0, 2))
        for m in ("CAM_ONLY", "AUTO", "CH0", "CH1", "CH2", "CH3"):
            tk.Radiobutton(
                tof_sel, text=m, variable=self.tof_mode_var, value=m,
                bg="#151c28", fg="#e2e8f0", selectcolor="#0f1520", activebackground="#151c28",
                font=("Consolas", 7, "bold"), command=lambda: self._push_tuning(force_arm=False)
            ).pack(side=tk.LEFT, padx=1)

        # WASD Keyboard Teleop & Restart Mission Controls (Strictly Nonholonomic: No Strafe)
        nudge_frame = tk.Frame(col2, bg="#0f1520", highlightbackground="#00e5ff", highlightthickness=1, padx=4, pady=3)
        nudge_frame.pack(fill=tk.X, pady=(2, 0))
        top_wasd = tk.Frame(nudge_frame, bg="#0f1520")
        top_wasd.pack(fill=tk.X, pady=(0, 2))
        tk.Label(
            top_wasd, text="🎮 WASD KEYBOARD TELEOP (No Strafe vy=0):",
            fg="#00e5ff", bg="#0f1520", font=("Segoe UI", 7, "bold")
        ).pack(side=tk.LEFT)
        tk.Button(
            top_wasd, text="↻ RESTART [R]", bg="#0284c7", fg="#ffffff",
            activebackground="#0369a1", relief=tk.FLAT, font=("Segoe UI", 7, "bold"), padx=6, pady=1,
            command=self._restart_mission
        ).pack(side=tk.RIGHT)

        btn_row = tk.Frame(nudge_frame, bg="#0f1520")
        btn_row.pack(fill=tk.X)
        wasd_defs = [
            ("w", "[W] ▲ Fwd", "#00e676", lambda: send_gui_command("wasd", vx=float(self.tuning.get("v_max_mps", 0.24)), wz=0.0, duration=0.45)),
            ("s", "[S] ▼ Back", "#ffab00", lambda: send_gui_command("wasd", vx=-0.85 * float(self.tuning.get("v_max_mps", 0.24)), wz=0.0, duration=0.45)),
            ("a", "[A] ↺ Left", "#38bdf8", lambda: send_gui_command("wasd", vx=0.0, wz=float(self.tuning.get("align_wz_max", 0.50)), duration=0.45)),
            ("d", "[D] ↻ Right", "#38bdf8", lambda: send_gui_command("wasd", vx=0.0, wz=-float(self.tuning.get("align_wz_max", 0.50)), duration=0.45)),
        ]
        for k_char, txt, fg_col, cmd_fn in wasd_defs:
            b = tk.Button(
                btn_row, text=txt, bg="#1e293b", fg=fg_col,
                activebackground="#334155", relief=tk.FLAT, font=("Consolas", 7, "bold"), padx=4, pady=1,
                command=cmd_fn
            )
            b.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
            self.wasd_btns[k_char] = (b, fg_col)

        tk.Button(
            btn_row, text="[Space] ■", bg="#dc2626", fg="#ffffff",
            relief=tk.FLAT, font=("Consolas", 7, "bold"), padx=4, pady=1,
            command=lambda: send_gui_command("wasd", vx=0.0, wz=0.0, duration=0.0)
        ).pack(side=tk.LEFT, padx=1)

        # ---------------------------------------------------------------------
        # COLUMN 3 (Right, Responsive remainder): ToF, Telemetry, Voltages & Table 1
        # ---------------------------------------------------------------------
        col3 = tk.Frame(body, bg="#0b0f17")
        col3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))

        # 1. Front ToF Sensors Auto-Discovery Card
        tof_card = tk.Frame(col3, bg="#151c28", highlightbackground="#283347", highlightthickness=1, padx=6, pady=4)
        tof_card.pack(fill=tk.X, pady=(0, 4))
        tk.Label(
            tof_card, text="FRONT ToF SENSORS AUTO-DISCOVERY (BLOCKED BY ARM vs. STRAIGHT CLEAR)",
            fg="#00e5ff", bg="#151c28", font=("Segoe UI", 7, "bold")
        ).pack(anchor="w", pady=(0, 2))

        tof_grid = tk.Frame(tof_card, bg="#151c28")
        tof_grid.pack(fill=tk.X)
        self.tof_labels = []
        for i in range(4):
            box = tk.Frame(tof_grid, bg="#0f1520", highlightbackground="#283347", highlightthickness=1, padx=4, pady=2)
            box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            lbl_title = tk.Label(box, text=f"ToF CH{i}", fg="#94a3b8", bg="#0f1520", font=("Consolas", 7, "bold"))
            lbl_title.pack()
            lbl_val = tk.Label(box, text="0 mm", fg="#e2e8f0", bg="#0f1520", font=("Consolas", 10, "bold"))
            lbl_val.pack()
            lbl_role = tk.Label(box, text="UNUSED", fg="#64748b", bg="#0f1520", font=("Consolas", 7, "bold"))
            lbl_role.pack()
            self.tof_labels.append((box, lbl_title, lbl_val, lbl_role))

        # 2. Real-Time Kinematics & Vision Grid Card
        metrics_card = tk.Frame(col3, bg="#151c28", highlightbackground="#283347", highlightthickness=1, padx=6, pady=4)
        metrics_card.pack(fill=tk.X, pady=(0, 4))
        tk.Label(
            metrics_card, text="REAL-TIME KINEMATICS (NO STRAFE vy=0) & VISION TELEMETRY",
            fg="#94a3b8", bg="#151c28", font=("Segoe UI", 7, "bold")
        ).pack(anchor="w", pady=(0, 2))

        m_grid = tk.Frame(metrics_card, bg="#151c28")
        m_grid.pack(fill=tk.X)
        self.kv_vars = {}
        fields = [
            ("mode", "Connection & Target"),
            ("sweep", "360° Sweep Progress"),
            ("dist", "Fused Dist / Stop Goal"),
            ("tof_clear", "Clear Front ToF Dist"),
            ("bearing", "Tag Bearing Angle"),
            ("skew", "Parallel Skew Angle"),
            ("vel", "Linear vx / vy (No Strafe)"),
            ("wz", "Turn Rate wz"),
            ("odom", "Odometry (x, y, yaw)"),
            ("mem", "Remembered Tags (360°)"),
        ]
        for idx, (key, title) in enumerate(fields):
            r, c = divmod(idx, 2)
            cell = tk.Frame(m_grid, bg="#0f1520", highlightbackground="#283347", highlightthickness=1, padx=4, pady=1)
            cell.grid(row=r, column=c, sticky="nsew", padx=2, pady=1)
            m_grid.columnconfigure(c, weight=1)
            tk.Label(cell, text=title, fg="#94a3b8", bg="#0f1520", font=("Segoe UI", 7)).pack(anchor="w")
            v_lbl = tk.Label(cell, text="-", fg="#e2e8f0", bg="#0f1520", font=("Consolas", 8, "bold"))
            v_lbl.pack(anchor="w")
            self.kv_vars[key] = v_lbl

        # 3. Motor Voltages V1..V4 & Wheel Speeds w1..w4 Card
        volt_card = tk.Frame(col3, bg="#151c28", highlightbackground="#283347", highlightthickness=1, padx=6, pady=4)
        volt_card.pack(fill=tk.X, pady=(0, 4))
        tk.Label(
            volt_card, text="CHASSIS MOTOR ARMATURE VOLTAGES & WHEEL SPEEDS (V = Kv*w + Ka*dw/dt)",
            fg="#94a3b8", bg="#151c28", font=("Segoe UI", 7, "bold")
        ).pack(anchor="w", pady=(0, 2))

        v_grid = tk.Frame(volt_card, bg="#151c28")
        v_grid.pack(fill=tk.X)
        self.volt_labels = []
        self.wheel_labels = []
        for i, mname in enumerate(("V1 (FR)", "V2 (FL)", "V3 (RL)", "V4 (RR)")):
            vbox = tk.Frame(v_grid, bg="#0f1520", highlightbackground="#283347", highlightthickness=1, padx=4, pady=2)
            vbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            tk.Label(vbox, text=mname, fg="#94a3b8", bg="#0f1520", font=("Consolas", 7, "bold")).pack()
            vl = tk.Label(vbox, text="0.000 V", fg="#00e676", bg="#0f1520", font=("Consolas", 9, "bold"))
            vl.pack()
            wl = tk.Label(vbox, text="w = +0.00 rad/s", fg="#38bdf8", bg="#0f1520", font=("Consolas", 7))
            wl.pack()
            self.volt_labels.append(vl)
            self.wheel_labels.append(wl)

        # 3b. Trajectory Matrices & Closed-Form Equations Card
        traj_card = tk.Frame(col3, bg="#151c28", highlightbackground="#283347", highlightthickness=1, padx=6, pady=3)
        traj_card.pack(fill=tk.X, pady=(0, 4))

        traj_top = tk.Frame(traj_card, bg="#151c28")
        traj_top.pack(fill=tk.X)
        tk.Label(
            traj_top, text="📐 DIFFERENTIAL FLATNESS CUBIC EQUATIONS & MATRICES",
            fg="#00e5ff", bg="#151c28", font=("Segoe UI", 7, "bold")
        ).pack(side=tk.LEFT)

        tk.Button(
            traj_top, text="📄 View Full Matrices", bg="#0284c7", fg="#ffffff",
            activebackground="#0369a1", font=("Segoe UI", 7, "bold"), relief=tk.FLAT, padx=6, pady=1,
            command=self._show_matrix_window
        ).pack(side=tk.RIGHT)

        poly_box = tk.Frame(traj_card, bg="#0f1520", highlightbackground="#283347", highlightthickness=1, padx=4, pady=2)
        poly_box.pack(fill=tk.X, pady=(2, 0))

        self.lbl_traj_x = tk.Label(
            poly_box, text="x(t) = +0.0000 +0.1600*t +0.0000*t² -0.0000*t³",
            fg="#00e676", bg="#0f1520", font=("Consolas", 8, "bold"), anchor="w"
        )
        self.lbl_traj_x.pack(fill=tk.X)

        self.lbl_traj_y = tk.Label(
            poly_box, text="y(t) = +0.0000 +0.0000*t +0.0000*t² -0.0000*t³",
            fg="#38bdf8", bg="#0f1520", font=("Consolas", 8), anchor="w"
        )
        self.lbl_traj_y.pack(fill=tk.X)

        self.lbl_traj_meta = tk.Label(
            poly_box, text="M(T) [4x4] Inverted | T = 3.50s | J = [[20, 4], [20, -4]] | Kv=0.380, Ka=0.060",
            fg="#94a3b8", bg="#0f1520", font=("Consolas", 7), anchor="w"
        )
        self.lbl_traj_meta.pack(fill=tk.X)

        # 4. Table 1 Motor Voltage Log (with Scrollbar and full expansion)
        t1_card = tk.Frame(col3, bg="#151c28", highlightbackground="#283347", highlightthickness=1, padx=6, pady=4)
        t1_card.pack(fill=tk.BOTH, expand=True)
        tk.Label(
            t1_card, text="TABLE 1: 1-SECOND MOTOR VOLTAGE TELEMETRY LOG (motor_voltage_log.csv)",
            fg="#94a3b8", bg="#151c28", font=("Segoe UI", 7, "bold")
        ).pack(anchor="w", pady=(0, 2))

        text_container = tk.Frame(t1_card, bg="#0f1520")
        text_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.t1_text = tk.Text(
            text_container, bg="#0f1520", fg="#e2e8f0", font=("Consolas", 8),
            relief=tk.FLAT, state=tk.DISABLED, yscrollcommand=scrollbar.set
        )
        self.t1_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.t1_text.yview)

    def _bind_keyboard(self):
        self.root.bind_all("<KeyPress>", self._on_key_press)
        self.root.bind_all("<KeyRelease>", self._on_key_release)

    def _toggle_cv2_window(self):
        self.show_cv2_window = not self.show_cv2_window
        if not self.show_cv2_window:
            try:
                cv2.destroyWindow("RoboMaster EP - Pure Native Feed")
            except Exception:
                pass
            self.btn_popout_cv2.configure(text="🗖 Pop-out Native Video (OpenCV)", fg="#38bdf8")
        else:
            self.btn_popout_cv2.configure(text="✖ Close Native Window", fg="#ffab00")

    def _on_action_button_click(self):
        if self.current_state in ("TELEOP", "WAITING"):
            self._start_mission()
        elif self.current_state in ("SEARCHING", "ALIGNING", "CHASING"):
            self._return_to_teleop()
        elif "ARRIVED" in self.current_state:
            self._return_to_teleop()
        elif "PAUSED" in self.current_state:
            if self.runner is not None:
                self.runner.start_mission()
            else:
                send_gui_command("resume")
        else:
            self._start_mission()

    def _start_mission(self):
        self.pressed_keys.clear()
        self._was_wasd_moving = False
        self._update_wasd_button_highlights()
        if self.runner is not None:
            self.runner.start_mission()
        else:
            send_gui_command("start_mission", target_tag_id=self.tuning.get("target_tag_id", 3))
        self.current_state = "SEARCHING"
        self.lbl_state.configure(text="⚡ STARTING MISSION...", fg="#00e5ff")
        self.btn_mission_action.configure(text="🛑 STOP / CANCEL MISSION [Space]", bg="#dc2626")

    def _return_to_teleop(self):
        self.pressed_keys.clear()
        self._was_wasd_moving = False
        self._update_wasd_button_highlights()
        if self.runner is not None:
            self.runner.return_to_teleop()
        else:
            send_gui_command("return_to_teleop")
        self.current_state = "TELEOP"
        self.lbl_state.configure(text="🟢 TELEOP / POSITIONING", fg="#00e676")
        self.btn_mission_action.configure(text="🚀 START AUTONOMOUS MISSION [Space]", bg="#059669")

    def _on_key_press(self, event):
        k = str(event.keysym).lower()
        if k == "r":
            self._restart_mission()
            return
        if k in ("space", "return"):
            self.pressed_keys.clear()
            self._was_wasd_moving = False
            self._on_action_button_click()
            self._update_wasd_button_highlights()
            return
        if k in ("escape", "x"):
            self.pressed_keys.clear()
            self._was_wasd_moving = False
            if self.runner is not None:
                self.runner.manual_drive(0.0, 0.0, 0.0)
            else:
                send_gui_command("estop")
            self._update_wasd_button_highlights()
            return
        key_map = {"w": "w", "up": "w", "s": "s", "down": "s", "a": "a", "left": "a", "d": "d", "right": "d"}
        if k in key_map:
            self.pressed_keys.add(key_map[k])
            self._send_wasd_from_keys()
            self._update_wasd_button_highlights()

    def _on_key_release(self, event):
        k = str(event.keysym).lower()
        key_map = {"w": "w", "up": "w", "s": "s", "down": "s", "a": "a", "left": "a", "d": "d", "right": "d"}
        if k in key_map:
            self.pressed_keys.discard(key_map[k])
            self._send_wasd_from_keys()
            self._update_wasd_button_highlights()

    def _update_wasd_button_highlights(self):
        for k_char, (btn, fg_col) in self.wasd_btns.items():
            if k_char in self.pressed_keys:
                btn.configure(bg="#0284c7", fg="#ffffff")
            else:
                btn.configure(bg="#1e293b", fg=fg_col)

    def _send_wasd_from_keys(self):
        v_max = float(self.tuning.get("v_max_mps", 0.28))
        w_max = float(self.tuning.get("align_wz_max", 0.50))
        vx = 0.0
        wz = 0.0
        if "w" in self.pressed_keys and "s" not in self.pressed_keys:
            vx = v_max
        elif "s" in self.pressed_keys and "w" not in self.pressed_keys:
            vx = -0.85 * v_max
        if "a" in self.pressed_keys and "d" not in self.pressed_keys:
            wz = w_max
        elif "d" in self.pressed_keys and "a" not in self.pressed_keys:
            wz = -w_max

        if abs(vx) > 1e-4 or abs(wz) > 1e-4:
            self._was_wasd_moving = True
            if self.runner is not None:
                self.runner.manual_drive(vx=vx, wz=wz, duration=0.40)
            else:
                send_gui_command("wasd", vx=vx, wz=wz, duration=0.40)
        elif self._was_wasd_moving:
            self._was_wasd_moving = False
            if self.runner is not None:
                self.runner.manual_drive(vx=0.0, wz=0.0, duration=0.0)
            else:
                send_gui_command("wasd", vx=0.0, wz=0.0, duration=0.0)

    def _restart_mission(self):
        self.pressed_keys.clear()
        self._was_wasd_moving = False
        self._update_wasd_button_highlights()
        send_gui_command("restart", target_tag_id=self.tuning.get("target_tag_id", 3))
        self.lbl_sync.configure(text=f"↻ Restart @ {time.strftime('%H:%M:%S')}", fg="#00e676")

    def _on_slider_drag(self, key: str):
        var, fmt_fn, is_int = self.slider_vars[key]
        val = int(round(var.get())) if is_int else round(float(var.get()), 3)
        self.slider_val_labels[key].configure(text=fmt_fn(val))
        self.tuning[key] = val

    def _push_tuning(self, force_arm: bool = False):
        for key, (var, _, is_int) in self.slider_vars.items():
            self.tuning[key] = int(round(var.get())) if is_int else round(float(var.get()), 3)
        self.tuning["tof_mode"] = self.tof_mode_var.get()
        save_tuning_and_notify(self.tuning, force_arm_move=force_arm)
        self.lbl_sync.configure(text=f"✓ Applied @ {time.strftime('%H:%M:%S')}", fg="#00e5ff")

    def _apply_preset(self, preset: dict):
        for k, v in preset.items():
            if k in self.slider_vars:
                var, fmt_fn, is_int = self.slider_vars[k]
                var.set(v)
                self.slider_val_labels[k].configure(text=fmt_fn(v))
                self.tuning[k] = int(round(v)) if is_int else round(float(v), 3)
        self._push_tuning(force_arm=False)

    def _set_arm_pose(self, x_mm: int, y_mm: int):
        self._apply_preset({"arm_x_mm": x_mm, "arm_y_mm": y_mm})
        self._push_tuning(force_arm=True)

    def _set_target_tag(self, tid: int):
        self.tuning["target_tag_id"] = int(tid)
        save_tuning_and_notify(self.tuning, force_arm_move=False)
        send_gui_command("set_tag", target_tag_id=tid)

    def _show_matrix_window(self):
        win = tk.Toplevel(self.root)
        win.title("Assessment 2 — Boundary Value Trajectory Matrices & Equations")
        win.geometry("820x640")
        win.configure(bg="#0b0f17")

        top_f = tk.Frame(win, bg="#151c28", padx=10, pady=8)
        top_f.pack(fill=tk.X)
        tk.Label(
            top_f, text="📐 DIFFERENTIAL FLATNESS BOUNDARY MATRICES & CLOSED-FORM EQUATIONS",
            fg="#00e5ff", bg="#151c28", font=("Segoe UI", 10, "bold")
        ).pack(side=tk.LEFT)
        tk.Button(
            top_f, text="✕ Close", bg="#dc2626", fg="#ffffff", font=("Segoe UI", 8, "bold"),
            relief=tk.FLAT, padx=8, pady=2, command=win.destroy
        ).pack(side=tk.RIGHT)

        txt_frame = tk.Frame(win, bg="#0f1520", padx=6, pady=6)
        txt_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        scroll = tk.Scrollbar(txt_frame, orient=tk.VERTICAL)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        txt = tk.Text(
            txt_frame, bg="#0f1520", fg="#00e5ff", font=("Consolas", 9),
            relief=tk.FLAT, yscrollcommand=scroll.set, padx=8, pady=8
        )
        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=txt.yview)

        meta = self._last_trajectory_analysis
        if meta and "boundary_matrix_M" in meta:
            report_str = format_trajectory_report(meta)
        else:
            meta = compute_trajectory_matrices(1.50, 0.0, 0.0)
            report_str = format_trajectory_report(meta)

        txt.insert(tk.END, report_str)
        txt.configure(state=tk.DISABLED)

    def _draw_radar_and_path(self, d: dict):
        c = self.map_canvas
        c.delete("all")
        w = max(380, c.winfo_width())
        h = max(160, c.winfo_height())

        # Left half: 360-Degree Tag Memory Radar
        rcx, rcy, rad = 75, h // 2, min(58, h // 2 - 12)
        c.create_oval(rcx - rad, rcy - rad, rcx + rad, rcy + rad, fill="#131b2e", outline="#334155", width=1)
        c.create_oval(rcx - rad // 2, rcy - rad // 2, rcx + rad // 2, rcy + rad // 2, outline="#1e293b", width=1)
        c.create_text(rcx, rcy - rad - 7, text="360° Radar", fill="#00e5ff", font=("Segoe UI", 7, "bold"))

        odom = d.get("odom", [0.0, 0.0, 0.0])
        yaw_rad = math.radians(float(odom[2]))
        hx = rcx + rad * 0.75 * math.cos(-yaw_rad)
        hy = rcy + rad * 0.75 * math.sin(-yaw_rad)
        c.create_line(rcx, rcy, hx, hy, fill="#00e5ff", width=2, arrow=tk.LAST)

        mem = d.get("tag_memory", {})
        tgt_id = str(d.get("target_tag_id", 3))
        for tid, info in mem.items():
            tyaw = math.radians(float(info.get("yaw_deg", 0.0)))
            tdist = min(0.92, max(0.20, float(info.get("dist_m", 1.2)) / 1.80))
            tx = rcx + rad * tdist * math.cos(-tyaw)
            ty = rcy + rad * tdist * math.sin(-tyaw)
            col = "#00e676" if str(tid) == tgt_id else "#ffab00"
            c.create_oval(tx - 4, ty - 4, tx + 4, ty + 4, fill=col, outline="#ffffff")
            c.create_text(tx + 12, ty, text=f"T{tid}", fill=col, font=("Consolas", 7, "bold"))

        # Right half: 2D Flat Cubic Trajectory & Persistent Odometry Trail Plot
        px0, py0, pw, ph = 160, 16, w - 170, h - 24
        c.create_rectangle(px0, py0, px0 + pw, py0 + ph, fill="#131b2e", outline="#334155")
        c.create_text(px0 + pw // 2, py0 - 7, text="2D Map: Flat Cubic z(t) (Cyan) vs Followed Path (Green)", fill="#00e5ff", font=("Segoe UI", 7, "bold"))

        # Gather points for dynamic bounding box calculation (never clip points!)
        pts_x = [0.0, 1.50]
        pts_y = [0.0, 0.0]
        pts_x.append(float(odom[0]))
        pts_y.append(float(odom[1]))

        path_pts = d.get("planned_path", [])
        for pt in path_pts:
            pts_x.append(pt[0])
            pts_y.append(pt[1])

        trail = d.get("odom_trail", [])
        if trail:
            self.current_trail = trail
            for pt in trail:
                pts_x.append(pt[0])
                pts_y.append(pt[1])

        hist = d.get("history_trails", [])
        if hist:
            self.history_trails = hist
            for htrail in hist:
                for pt in htrail:
                    pts_x.append(pt[0])
                    pts_y.append(pt[1])

        min_x = min(min(pts_x) - 0.20, -0.20)
        max_x = max(max(pts_x) + 0.25, 1.75)
        span_x = max(max_x - min_x, 1.5)

        min_y = min(min(pts_y) - 0.25, -0.55)
        max_y = max(max(pts_y) + 0.25, 0.55)
        span_y = max(max_y - min_y, 0.90)

        def to_canvas(wx, wy):
            u = px0 + 14 + ((wx - min_x) / span_x) * (pw - 28)
            v = py0 + ph - 14 - ((wy - min_y) / span_y) * (ph - 28)
            return u, v

        # Draw grid lines & axes
        ox0, oy0 = to_canvas(0.0, 0.0)
        c.create_line(px0 + 4, oy0, px0 + pw - 4, oy0, fill="#1e293b", dash=(2, 2))
        c.create_line(ox0, py0 + 4, ox0, py0 + ph - 4, fill="#1e293b", dash=(2, 2))

        # Draw Tag T3 (1.50m) and Standoff (1.30m)
        tag_x, tag_y = to_canvas(1.50, 0.0)
        c.create_rectangle(tag_x - 5, tag_y - 8, tag_x + 5, tag_y + 8, fill="#ef4444", outline="#ffffff")
        c.create_text(tag_x, tag_y - 13, text=f"Tag {tgt_id} (1.5m)", fill="#ef4444", font=("Segoe UI", 6, "bold"))

        goal_x, goal_y = to_canvas(1.30, 0.0)
        c.create_line(goal_x, goal_y - 10, goal_x, goal_y + 10, fill="#10b981", width=2, dash=(2, 2))
        c.create_text(goal_x, goal_y + 13, text="Goal (20cm)", fill="#10b981", font=("Segoe UI", 6))

        # 1. Draw Past History Trails (subtle dark green, persistent)
        for htrail in self.history_trails:
            if len(htrail) >= 2:
                h_coords = []
                for pt in htrail:
                    u, v = to_canvas(pt[0], pt[1])
                    h_coords.extend([u, v])
                c.create_line(*h_coords, fill="#1e4a38", width=1.5, dash=(3, 2))

        # 2. Draw Planned Flat Cubic Path z(t) (bright cyan, dashed)
        if len(path_pts) >= 2:
            flat_coords = []
            for pt in path_pts:
                u, v = to_canvas(pt[0], pt[1])
                flat_coords.extend([u, v])
            c.create_line(*flat_coords, fill="#00e5ff", width=2, dash=(3, 2))

        # 3. Draw Active Followed Path (vibrant neon green, solid, persistent)
        active_trail = self.current_trail if len(self.current_trail) >= 2 else trail
        if len(active_trail) >= 2:
            t_coords = []
            for pt in active_trail:
                u, v = to_canvas(pt[0], pt[1])
                t_coords.extend([u, v])
            c.create_line(*t_coords, fill="#00e676", width=2.5)

        # 4. Draw Start Pose (0, 0)
        c.create_oval(ox0 - 3, oy0 - 3, ox0 + 3, oy0 + 3, fill="#38bdf8", outline="#ffffff")

        # 5. Draw Robot Current Pose
        rx, ry = to_canvas(float(odom[0]), float(odom[1]))
        r_heading_x = rx + 10 * math.cos(-yaw_rad)
        r_heading_y = ry + 10 * math.sin(-yaw_rad)
        c.create_oval(rx - 5, ry - 5, rx + 5, ry + 5, fill="#00e676", outline="#ffffff", width=1.5)
        c.create_line(rx, ry, r_heading_x, r_heading_y, fill="#ffffff", width=2, arrow=tk.LAST)

    def _save_path_plot(self):
        """Generates a publication-quality 2D path map of the followed trajectory and saves it."""
        try:
            import datetime
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            meta = self._last_trajectory_analysis or compute_trajectory_matrices(1.50, 0.0, 0.0)
            alpha_x = np.array(meta.get("coefficients_alpha_x", [0.0, 0.06, 0.11, -0.01]))
            alpha_y = np.array(meta.get("coefficients_alpha_y", [0.0, 0.0, 0.0, 0.0]))
            T_plan = float(meta.get("T_s", 3.5))
            d0 = float(meta.get("goal_pose", {}).get("d0_m", 1.50))
            standoff = float(meta.get("goal_pose", {}).get("standoff_m", 0.20))
            xf = float(meta.get("goal_pose", {}).get("xf_m", d0 - standoff))

            t_eval = np.linspace(0.0, T_plan, 200)
            x_plan = alpha_x[0] + alpha_x[1] * t_eval + alpha_x[2] * (t_eval**2) + alpha_x[3] * (t_eval**3)
            y_plan = alpha_y[0] + alpha_y[1] * t_eval + alpha_y[2] * (t_eval**2) + alpha_y[3] * (t_eval**3)

            trail = self.current_trail
            if not trail and self.runner:
                trail = getattr(self.runner, "odom_trail", [])

            fig, ax = plt.subplots(figsize=(8.5, 4.2))

            ax.axvline(d0, color='darkred', linestyle='--', lw=1.5, alpha=0.7, label=f'Target Tag Position (x = {d0:.2f} m)')
            ax.axvspan(xf, d0, color='red', alpha=0.08, label=f'{int(standoff*100)} cm Standoff Safety Buffer')
            ax.plot(d0, 0.0, 's', color='darkred', markersize=9, markeredgecolor='black', label='Target Tag Center')
            ax.plot(xf, 0.0, 'X', color='darkgreen', markersize=9, markeredgecolor='black', label=f'Standoff Goal (x = {xf:.2f} m)')

            ax.plot(x_plan, y_plan, color='#1565c0', lw=2.5, linestyle='-', label=r'Planned Cubic Flat Path $\mathbf{z}(t)$')

            if trail and len(trail) >= 2:
                tx = [p[0] for p in trail]
                ty = [p[1] for p in trail]
                ax.plot(tx, ty, color='#2e7d32', lw=2.2, linestyle='--', marker='o', markevery=max(1, len(tx)//12), markersize=4.5, label='Actual Followed Path (Hardware)')

            for idx, htrail in enumerate(self.history_trails):
                if len(htrail) >= 2:
                    hx = [p[0] for p in htrail]
                    hy = [p[1] for p in htrail]
                    lbl = 'Previous Run Trail' if idx == 0 else None
                    ax.plot(hx, hy, color='#78909c', lw=1.2, linestyle=':', alpha=0.7, label=lbl)

            ax.plot(0.0, 0.0, 'o', color='black', markersize=8, label='Start Pose $(0, 0)$')

            poly_box = (
                f"Differential Flatness Cubic Model:\n"
                f"x(t) = {alpha_x[0]:.2f} + {alpha_x[1]:.2f}t + {alpha_x[2]:+.3f}t² + {alpha_x[3]:+.4f}t³\n"
                f"T = {T_plan:.2f} s | Standoff = {standoff:.2f} m"
            )
            ax.text(
                0.03, 0.95, poly_box, transform=ax.transAxes, fontsize=8.2, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4fd', edgecolor='#0284c7', alpha=0.92)
            )

            ax.set_xlabel('Arena Longitudinal Position $X$ [m]')
            ax.set_ylabel('Arena Lateral Position $Y$ [m]')
            ax.set_title('2D Path Map: Planned Differential Flatness vs. Actual Followed Path', pad=8)
            ax.set_xlim(-0.15, d0 + 0.25)
            ax.set_ylim(-0.40, 0.40)
            ax.grid(True)
            ax.legend(loc='lower left', framealpha=0.92, ncol=2)
            plt.tight_layout()

            os.makedirs(os.path.join(BASE_DIR, "runs"), exist_ok=True)
            tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(BASE_DIR, "runs", f"path_2d_saved_{tstamp}.png")
            plt.savefig(save_path, dpi=300)
            plt.savefig(os.path.join(BASE_DIR, "runs", "latest_path_2d.png"), dpi=300)
            rep_path = os.path.join(BASE_DIR, "report", "fig_path_2d_planned_vs_actual.png")
            if os.path.exists(os.path.dirname(rep_path)):
                plt.savefig(rep_path, dpi=300)
            plt.close()

            print(f"[+] Saved 2D Path Map plot to: {save_path}")
            if hasattr(self, 'btn_save_path'):
                self.btn_save_path.configure(text="✔ Saved!", bg="#059669")
                self.root.after(2000, lambda: self.btn_save_path.configure(text="💾 Save Path Plot", bg="#0284c7"))
        except Exception as e:
            print(f"[!] Error saving path plot: {e}")

    def _update_loop(self):
        if self.pressed_keys:
            self._send_wasd_from_keys()

        # ---------------------------------------------------------------------
        # 1. Update Live Camera Frame (Zero Latency Direct Stream)
        # ---------------------------------------------------------------------
        got_frame = False
        if self.runner is not None and getattr(self.runner, "latest_raw_frame", None) is not None:
            try:
                raw_frame = self.runner.latest_raw_frame
                # Pure native display: No processing, instant in-memory blit (<2.5ms)
                rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                if pil_img.size != (380, 214):
                    pil_img = pil_img.resize((380, 214), Image.NEAREST)
                self._tk_cam_img = ImageTk.PhotoImage(pil_img)
                self.cam_label.configure(image=self._tk_cam_img)
                got_frame = True

                self._frame_count += 1
                t_now = time.monotonic()
                if (t_now - self._fps_time) >= 1.0:
                    self._fps_val = self._frame_count / (t_now - self._fps_time)
                    self._frame_count = 0
                    self._fps_time = t_now
                    self.lbl_cam_fps.configure(
                        text=f"{raw_frame.shape[1]}x{raw_frame.shape[0]} | {self._fps_val:.0f} FPS Native",
                        fg="#00e676"
                    )

                if self.show_cv2_window:
                    cv2.imshow("RoboMaster EP - Pure Native Feed", raw_frame)
                    cv2.waitKey(1)
            except Exception:
                pass

        elif os.path.exists(IMG_PATH):
            try:
                with open(IMG_PATH, "rb") as f:
                    raw_data = f.read()
                if raw_data and len(raw_data) > 500:
                    pil_img = Image.open(io.BytesIO(raw_data))
                    if pil_img.size != (380, 214):
                        pil_img = pil_img.resize((380, 214), Image.NEAREST)
                    self._tk_cam_img = ImageTk.PhotoImage(pil_img)
                    self.cam_label.configure(image=self._tk_cam_img)
                    got_frame = True
            except Exception:
                pass

        if not got_frame and self._tk_cam_img is None:
            blank = Image.new("RGB", (380, 214), (15, 21, 32))
            draw = ImageDraw.Draw(blank)
            draw.text((35, 100), "Waiting for Camera Stream (Zero Latency)...", fill=(0, 229, 255))
            self._tk_cam_img = ImageTk.PhotoImage(blank)
            self.cam_label.configure(image=self._tk_cam_img)

        # ---------------------------------------------------------------------
        # 2. Update Telemetry & State (Direct RAM if unified runner, else JSON)
        # ---------------------------------------------------------------------
        d = None
        if self.runner is not None:
            try:
                d = self.runner.get_telemetry_dict()
            except Exception:
                d = None

        if d is None and os.path.exists(JSON_PATH):
            try:
                with open(JSON_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                d = None

        if d is not None:
            try:
                st = d.get("state", "IDLE")
                self.current_state = st
                ts = d.get("timestamp", 0.0)

                if st == "TELEOP":
                    st_col = "#00e676"
                    disp_text = f"🟢 TELEOP / POSITIONING | t={ts:.1f}s"
                    self.btn_mission_action.configure(text="🚀 START AUTONOMOUS MISSION [Space]", bg="#059669")
                elif "WASD" in st:
                    st_col = "#00e5ff"
                    disp_text = f"🎮 TELEOP DRIVING | t={ts:.1f}s"
                    self.btn_mission_action.configure(text="🚀 START AUTONOMOUS MISSION [Space]", bg="#059669")
                elif "ARRIVED" in st:
                    st_col = "#00e676"
                    disp_text = f"🏁 ARRIVED (20cm) | t={ts:.1f}s"
                    self.btn_mission_action.configure(text="🔄 RETURN TO TELEOP (RE-POSITION) [Space]", bg="#0284c7")
                elif "PAUSED" in st:
                    st_col = "#ff5252"
                    disp_text = f"🛑 PAUSED (E-STOP) | t={ts:.1f}s"
                    self.btn_mission_action.configure(text="▶ RESUME MISSION [Space]", bg="#16a34a")
                else:
                    st_col = "#ffab00" if "SEARCH" in st else "#00e5ff"
                    disp_text = f"⚡ {st} | t={ts:.1f}s"
                    self.btn_mission_action.configure(text="🛑 STOP / CANCEL MISSION [Space]", bg="#dc2626")

                self.lbl_state.configure(text=disp_text, fg=st_col)

                ch_mm = d.get("tof_channels_mm", [0, 0, 0, 0])
                ch_st = d.get("tof_status", ["UNUSED"] * 4)
                act_ch = d.get("tof_active_ch", 0)
                for i in range(4):
                    box, _, l_val, l_role = self.tof_labels[i]
                    mm = ch_mm[i] if i < len(ch_mm) else 0
                    role = ch_st[i] if i < len(ch_st) else "UNUSED"
                    l_val.configure(text=f"{mm} mm")
                    if i == act_ch and role in ("STRAIGHT_CLEAR", "CLEAR_AUX"):
                        box.configure(highlightbackground="#00e676", highlightthickness=2)
                        l_role.configure(text="★ STRAIGHT CLEAR", fg="#00e676")
                    elif "BLOCKED" in role:
                        box.configure(highlightbackground="#ff5252", highlightthickness=2)
                        l_role.configure(text="✖ BLOCKED BY ARM", fg="#ff5252")
                    else:
                        box.configure(highlightbackground="#283347", highlightthickness=1)
                        l_role.configure(text=role, fg="#64748b")

                self.kv_vars["mode"].configure(text=f"{d.get('mode', '-')} | Tag {d.get('target_tag_id', 3)}")
                self.kv_vars["sweep"].configure(text=f"{d.get('sweep_deg', 0.0):.1f}° / 360.0°")
                self.kv_vars["dist"].configure(
                    text=f"{d.get('fused_dist_m', 0.0):.3f} m (Stop: {d.get('standoff_goal_m', 0.20):.2f} m)",
                    fg="#00e676"
                )
                cam_d = float(d.get('cam_dist_m', 0.0))
                tof_d = float(d.get('tof_dist_m', 0.0))
                fused_d = float(d.get('fused_dist_m', 0.0))
                if all(m == 0 for m in ch_mm):
                    self.kv_vars["tof_clear"].configure(
                        text=f"PURE VISION: {fused_d:.2f}m (Cam: {cam_d:.2f}m)", fg="#00e5ff"
                    )
                else:
                    self.kv_vars["tof_clear"].configure(
                        text=f"ToF:{tof_d:.2f}m (CH{act_ch}) | Cam:{cam_d:.2f}m"
                    )
                vis_str = "VISIBLE" if d.get("tag_visible") else "MEMORY"
                self.kv_vars["bearing"].configure(text=f"{d.get('bearing_deg', 0.0):+.1f}° ({vis_str})")
                self.kv_vars["skew"].configure(text=f"{d.get('skew_deg', 0.0):+.1f}° (0°=Parallel)")
                self.kv_vars["vel"].configure(text=f"vx={d.get('vx', 0.0):+.3f} m/s, vy=0.000 (NO STRAFE)")
                self.kv_vars["wz"].configure(text=f"{d.get('wz_rad', 0.0):+.2f} rad/s ({d.get('wz_deg', 0.0):+.1f}°/s)")
                odom = d.get("odom", [0, 0, 0])
                self.kv_vars["odom"].configure(
                    text=f"x={odom[0]:+.2f}m, y={odom[1]:+.2f}m, yaw={odom[2]:+.1f}°",
                    fg="#00e5ff"
                )

                mem = d.get("tag_memory", {})
                if mem:
                    m_txt = ", ".join([f"T{k}@{v['yaw_deg']:+.0f}°({v['dist_m']:.2f}m)" for k, v in mem.items()])
                else:
                    m_txt = "Sweeping 360°..."
                self.kv_vars["mem"].configure(text=m_txt, fg="#ffab00")

                volts = d.get("voltages", [0, 0, 0, 0])
                wrads = d.get("wheel_rads", [0, 0, 0, 0])
                for i in range(4):
                    v = float(volts[i]) if i < len(volts) else 0.0
                    w = float(wrads[i]) if i < len(wrads) else 0.0
                    v_col = "#00e676" if v > 0.02 else ("#ffab00" if v < -0.02 else "#94a3b8")
                    self.volt_labels[i].configure(text=f"{v:+.3f} V", fg=v_col)
                    self.wheel_labels[i].configure(text=f"ω = {w:+.2f} rad/s")

                rows = d.get("table1", [])
                self.t1_text.configure(state=tk.NORMAL)
                self.t1_text.delete("1.0", tk.END)
                header = f"{'t(s)':<7} {'V1 (FR)':>9} {'V2 (FL)':>9} {'V3 (RL)':>9} {'V4 (RR)':>9}   {'State'}\n"
                self.t1_text.insert(tk.END, header + "-" * 58 + "\n")
                for r in rows[-15:]:
                    line = f"{r[0]:<7} {float(r[1]):>+9.3f} {float(r[2]):>+9.3f} {float(r[3]):>+9.3f} {float(r[4]):>+9.3f}   {r[5]}\n"
                    self.t1_text.insert(tk.END, line)
                self.t1_text.see(tk.END)
                self.t1_text.configure(state=tk.DISABLED)

                # Update Trajectory Matrices & Equations Card
                t_meta = d.get("trajectory_analysis", {})
                if t_meta:
                    self._last_trajectory_analysis = t_meta
                    eqs = t_meta.get("equations", {})
                    if eqs:
                        self.lbl_traj_x.configure(text=eqs.get("x_t", ""))
                        self.lbl_traj_y.configure(text=eqs.get("y_t", ""))
                    T_val = t_meta.get("T_s", 0.0)
                    mp = t_meta.get("motor_params", {})
                    self.lbl_traj_meta.configure(
                        text=f"M(T) [4x4] Inverted | T = {T_val:.2f}s | J: [20, ±4] | Kv={mp.get('Kv', 0.380)}, Ka={mp.get('Ka', 0.060)}"
                    )

                self._draw_radar_and_path(d)
            except Exception:
                pass

        self.root.after(25, self._update_loop)


class WebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/camera.jpg") and os.path.exists(IMG_PATH):
            with open(IMG_PATH, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith("/telemetry.json") and os.path.exists(JSON_PATH):
            with open(JSON_PATH, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body style='background:#0b0f17;color:#fff;font-family:sans-serif'><h2>RoboMaster v4 Web Mirror Active</h2><img src='/camera.jpg' id='c'/><script>setInterval(()=>document.getElementById('c').src='/camera.jpg?t='+Date.now(),150)</script></body></html>")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DJI RoboMaster EP Real-Time Telemetry & GUI")
    parser.add_argument("--conn", choices=["ap", "rndis"], default="ap", help="Connection mode (default: ap)")
    parser.add_argument("--tag", type=int, default=3, help="Target AprilTag ID (default: 3)")
    parser.add_argument("--sim", action="store_true", help="Launch native simulation instead of physical robot")
    parser.add_argument("--external", action="store_true", help="Pure external listener mode (do not connect to hardware directly; listen to WSL/JSON)")
    parser.add_argument("--vmax", type=float, default=0.38, help="Max speed (m/s)")
    args = parser.parse_args()

    try:
        srv = HTTPServer(("127.0.0.1", 8080), WebHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    except Exception:
        pass

    runner = None
    if not args.external:
        try:
            from mission_runner_win import WindowsMissionRunner
            runner = WindowsMissionRunner(
                conn_type=args.conn,
                target_tag_id=args.tag,
                use_sim=args.sim,
                v_max=args.vmax,
                output_dir=BASE_DIR
            )
            def _runner_loop():
                while not runner._stop_evt.is_set():
                    runner.step_control()
                    time.sleep(0.048)
            threading.Thread(target=_runner_loop, daemon=True).start()
        except Exception as e:
            print(f"[GUI] Note: Direct hardware runner init ({e}). Running in external listener mode.")
            runner = None

    root = tk.Tk()
    app = DebugApp(root, runner=runner)

    def on_closing():
        if runner is not None:
            try:
                runner.cleanup()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
