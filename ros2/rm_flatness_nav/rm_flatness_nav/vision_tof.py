#!/usr/bin/env python3
"""
vision_tof.py - AprilTag Parallel Vision Detector & Dual Front ToF Auto-Selector (v4)
=====================================================================================
1. FrontToFSelector:
   - Parses all 4 channels of `sub_distance` (`[d0, d1, d2, d3]`).
   - Distinguishes the ToF sensor BLOCKED_BY_ARM (short, static reading invariant to
     robot rotation/motion) from the STRAIGHT_CLEAR ToF sensor (varies during sweep
     and tracks obstacles ahead).
   - Among unblocked channels, uses the closest valid reading so the robot never
     ignores an approaching obstacle/tag.
2. AprilTagDetector:
   - Detects AprilTags across the full camera frame (including between the gripper
     jaws at close range < 30 cm so the robot never goes blind near the tag).
   - Calibrated for the 60 mm printable AprilTags (48 mm inner black marker square)
     plus downward-camera floor elevation geometry so 20 cm standoff is accurate.
"""

import math
from collections import deque
import numpy as np
import cv2

from rm_flatness_nav.flatness_math import wrap_angle


class FrontToFSelector:
    """
    Identifies which front ToF sensor channel is blocked by the robotic arm
    and which channel looks straight ahead.
    """

    def __init__(self):
        self.channels_mm = [0, 0, 0, 0]
        self.status = ["UNUSED", "UNUSED", "UNUSED", "UNUSED"]
        self.history = [deque(maxlen=45) for _ in range(4)]
        self.min_seen_mm = [99999, 99999, 99999, 99999]
        self.max_seen_mm = [0, 0, 0, 0]
        self.active_idx = 0
        self.locked_idx = None
        self.blocked_thresh_mm = 220.0
        self.tof_mode = "AUTO"  # "AUTO", "CH0", "CH1", "CH2", "CH3"
        self.dist_m = 1.50
        self.last_update_time = 0.0

    def reset_sweep_stats(self):
        self.locked_idx = None
        self.min_seen_mm = [99999, 99999, 99999, 99999]
        self.max_seen_mm = [0, 0, 0, 0]

    def update(self, sub_info, now_s: float):
        if sub_info is None:
            return
        # Unwrap nested tuple/list if SDK passes ([d0, d1, d2, d3],)
        if isinstance(sub_info, (list, tuple)) and len(sub_info) == 1 and isinstance(sub_info[0], (list, tuple)):
            raw_list = sub_info[0]
        elif isinstance(sub_info, (list, tuple)):
            raw_list = sub_info
        else:
            return

        for i in range(min(4, len(raw_list))):
            try:
                val = int(float(raw_list[i])) if raw_list[i] is not None else 0
            except Exception:
                val = 0
            self.channels_mm[i] = val
            # Valid physical ToF range on RoboMaster EP is ~15 mm to 8000 mm (65534/65535 = out of range)
            if 15 <= val < 8000:
                self.history[i].append(val)
                if val < self.min_seen_mm[i]:
                    self.min_seen_mm[i] = val
                if val > self.max_seen_mm[i]:
                    self.max_seen_mm[i] = val

        thresh = float(self.blocked_thresh_mm)

        # Manual channel override from GUI if selected
        if self.tof_mode in ("CH0", "CH1", "CH2", "CH3"):
            forced = int(self.tof_mode[-1])
            self.active_idx = forced
            for i in range(4):
                if i == forced:
                    self.status[i] = "STRAIGHT_CLEAR"
                elif self.channels_mm[i] < 15 or self.channels_mm[i] >= 8000:
                    self.status[i] = "UNUSED"
                elif self.channels_mm[i] < thresh:
                    self.status[i] = "BLOCKED_BY_ARM"
                else:
                    self.status[i] = "CLEAR_AUX"
            active_mm = self.channels_mm[self.active_idx]
            if 15 <= active_mm < 8000:
                raw_m = float(active_mm) / 1000.0
                self.dist_m = 0.70 * raw_m + 0.30 * self.dist_m
                self.last_update_time = now_s
            return

        # Classify connected channels in AUTO mode
        connected = []
        for i in range(4):
            cur = self.channels_mm[i]
            hist = list(self.history[i])
            if not hist or cur < 15 or cur >= 8000:
                self.status[i] = "UNUSED"
                continue
            med = float(np.median(hist))
            p90 = float(np.percentile(hist, 90)) if len(hist) >= 4 else float(max(hist))
            p10 = float(np.percentile(hist, 10)) if len(hist) >= 4 else float(min(hist))
            variation = p90 - p10
            mx = self.max_seen_mm[i]
            connected.append({
                "idx": i,
                "cur": cur,
                "med": med,
                "var": variation,
                "max": mx,
            })

        if not connected:
            return

        if len(connected) >= 2:
            # Determine which sensor(s) are blocked by the arm:
            # An arm-blocked sensor never sees open room (max < thresh + 40) AND has very low variation (< 40 mm)
            # while sitting at short range (< thresh).
            clear_candidates = []
            for c in connected:
                is_arm_blocked = (c["max"] < (thresh + 35.0)) and (c["med"] < thresh) and (c["var"] < 40.0)
                if is_arm_blocked:
                    self.status[c["idx"]] = "BLOCKED_BY_ARM"
                else:
                    clear_candidates.append(c)

            # Fallback if both sensors looked short/static so far: the one with larger max/variation is clear,
            # the one with smaller static reading is blocked by the arm.
            if not clear_candidates:
                by_openness = sorted(connected, key=lambda c: (c["max"], c["var"], c["med"]), reverse=True)
                clear_candidates = [by_openness[0]]
                for c in by_openness[1:]:
                    self.status[c["idx"]] = "BLOCKED_BY_ARM"
            elif len(clear_candidates) == len(connected):
                # If 2 sensors are connected and one has never exceeded thresh while the other has seen open space (> 350 mm),
                # mark the short one as BLOCKED_BY_ARM.
                max_overall = max(c["max"] for c in connected)
                if max_overall > 350:
                    filtered = [c for c in clear_candidates if not (c["med"] < thresh and c["max"] < 300)]
                    if filtered:
                        for c in clear_candidates:
                            if c not in filtered:
                                self.status[c["idx"]] = "BLOCKED_BY_ARM"
                        clear_candidates = filtered

            # Among the unblocked clear sensor(s), pick the locked one OR the one reading the CLOSEST distance!
            if self.locked_idx is not None and any(c["idx"] == self.locked_idx for c in clear_candidates):
                best_i = self.locked_idx
            else:
                # Choose the unblocked sensor reading the closest obstacle (min cur_mm)
                best_c = min(clear_candidates, key=lambda c: c["cur"])
                best_i = best_c["idx"]

            for c in clear_candidates:
                self.status[c["idx"]] = "STRAIGHT_CLEAR" if c["idx"] == best_i else "CLEAR_AUX"
            self.active_idx = best_i

        else:
            # Single connected ToF sensor -> always use it as STRAIGHT_CLEAR
            c = connected[0]
            i = c["idx"]
            self.status[i] = "STRAIGHT_CLEAR"
            self.active_idx = i

        # Update smoothed distance in meters from the active straight ToF sensor
        active_mm = self.channels_mm[self.active_idx]
        if 15 <= active_mm < 8000 and self.status[self.active_idx] == "STRAIGHT_CLEAR":
            raw_m = float(active_mm) / 1000.0
            # Fast response when distance is dropping so braking at 20 cm is never delayed
            alpha = 0.85 if raw_m < self.dist_m else 0.65
            self.dist_m = alpha * raw_m + (1.0 - alpha) * self.dist_m
            self.last_update_time = now_s

    def lock_straight_sensor(self):
        """Locks onto the discovered STRAIGHT_CLEAR ToF channel after the 360 sweep."""
        if self.status[self.active_idx] == "STRAIGHT_CLEAR":
            self.locked_idx = self.active_idx


class AprilTagDetector:
    """
    Detects AprilTags on the floor/stand and computes:
      - id: AprilTag ID
      - dist_m: Camera-estimated distance to the tag (m)
      - bearing_rad: Horizontal angle to tag center (+left, -right)
      - skew_rad: Parallel alignment error from the 4 tag corners (0 = perfectly parallel)
    """

    def __init__(self, tag_size_m: float = 0.048):
        self.tag_size_m = float(tag_size_m)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.primary_detector = None
        self.aux_detectors = []

        if hasattr(cv2, "aruco"):
            did_36 = getattr(cv2.aruco, "DICT_APRILTAG_36h11", None)
            if did_36 is not None:
                adict = cv2.aruco.getPredefinedDictionary(did_36)
                if hasattr(cv2.aruco, "DetectorParameters"):
                    params = cv2.aruco.DetectorParameters()
                    params.minMarkerPerimeterRate = 0.012
                    self.primary_detector = cv2.aruco.ArucoDetector(adict, params)
                else:
                    params = cv2.aruco.DetectorParameters_create()
                    params.minMarkerPerimeterRate = 0.012
                    self.primary_detector = (adict, params)

            for name in ("DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_4X4_50"):
                did = getattr(cv2.aruco, name, None)
                if did is not None:
                    try:
                        adict = cv2.aruco.getPredefinedDictionary(did)
                        if hasattr(cv2.aruco, "DetectorParameters"):
                            p = cv2.aruco.DetectorParameters()
                            p.minMarkerPerimeterRate = 0.015
                            self.aux_detectors.append((name, cv2.aruco.ArucoDetector(adict, p)))
                        else:
                            p = cv2.aruco.DetectorParameters_create()
                            p.minMarkerPerimeterRate = 0.015
                            self.aux_detectors.append((name, (adict, p)))
                    except Exception:
                        pass

    @staticmethod
    def _run_detector(det, gray):
        if det is None:
            return None, None, None
        try:
            if isinstance(det, tuple):
                if hasattr(cv2.aruco, "detectMarkers"):
                    return cv2.aruco.detectMarkers(gray, det[0], parameters=det[1])
                return None, None, None
            return det.detectMarkers(gray)
        except Exception:
            return None, None, None

    def _parse_corners(self, corners, ids, fx: float, fy: float, cx_mid: float, h_img: int, scale: float = 1.0):
        if ids is None or len(ids) == 0:
            return []
        results = []
        seen = set()
        for i, raw_id in enumerate(np.asarray(ids).flatten()):
            tid = int(raw_id)
            if tid in seen:
                continue
            pts = corners[i][0].astype(np.float64) * scale
            cx = float(np.mean(pts[:, 0]))

            x1, y1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
            x2, y2 = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))
            bw, bh = max(4, x2 - x1), max(4, y2 - y1)

            dist_m = self._estimate_tag_distance(bw, bh, y2, h_img, fy)
            bearing_rad = -math.atan2(cx - cx_mid, fx)
            skew_rad = self.compute_parallel_skew(pts)

            seen.add(tid)
            results.append({
                "id": tid,
                "dist_m": dist_m,
                "bearing_rad": bearing_rad,
                "skew_rad": skew_rad,
                "bbox": (x1, y1, bw, bh),
                "corners": pts,
            })
        return results

    @staticmethod
    def compute_parallel_skew(pts: np.ndarray) -> float:
        """
        Computes how skewed/tilted the tag is relative to the horizontal camera frame.
        When skew_rad == 0, the tag fits squarely and parallelly in the camera view.
        """
        sy = pts[np.argsort(pts[:, 1])]
        tl, tr = sy[:2][np.argsort(sy[:2, 0])]
        bl, br = sy[2:][np.argsort(sy[2:, 0])]

        ang_top = math.atan2(float(tr[1] - tl[1]), max(abs(float(tr[0] - tl[0])), 1.0))
        ang_bot = math.atan2(float(br[1] - bl[1]), max(abs(float(br[0] - bl[0])), 1.0))
        edge_slope = 0.5 * (ang_top + ang_bot)

        h_left = float(np.linalg.norm(bl - tl))
        h_right = float(np.linalg.norm(br - tr))
        persp_diff = (h_left - h_right) / max(4.0, 0.5 * (h_left + h_right))

        return float(edge_slope + 0.35 * persp_diff)

    def _estimate_tag_distance(self, bw: float, bh: float, y_bottom: float, h_img: int, fy: float) -> float:
        """
        Estimates distance (m) to the AprilTag using calibrated pinhole optics:
        Z = (tag_size_m * fy) / apparent_px
        Using max(bw, bh) ensures that floor tilt / pitch foreshortening of vertical height bh
        does not cause distance overestimation.
        """
        apparent_px = max(float(bw), float(bh))
        size_dist = (self.tag_size_m * fy) / max(apparent_px, 6.0)
        return float(np.clip(size_dist, 0.08, 3.00))

    def detect_all(self, frame: np.ndarray, default_id: int = 3, check_aux: bool = False):
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h_img, w_img = gray.shape[:2]
        fx = (w_img / 640.0) * 520.0
        fy = (h_img / 360.0) * 390.0
        cx_mid = w_img / 2.0

        # Step 1: Ultra-fast primary AprilTag 36h11 detection (5-15 ms)
        corners, ids, _ = self._run_detector(self.primary_detector, gray)
        dets = self._parse_corners(corners, ids, fx, fy, cx_mid, h_img)
        if dets:
            return dets

        # Step 2: High-contrast 360p CLAHE pass (catches distant or dim tags in ~10 ms)
        g_small = cv2.resize(gray, (640, 360)) if (w_img != 640 or h_img != 360) else gray
        scale = float(w_img) / 640.0 if (w_img != 640) else 1.0
        g_enh = self.clahe.apply(g_small)
        corners_c, ids_c, _ = self._run_detector(self.primary_detector, g_enh)
        dets_c = self._parse_corners(corners_c, ids_c, fx, fy, cx_mid, h_img, scale=scale)
        if dets_c:
            return dets_c

        # Step 3: Optional aux detector check (only if explicitly requested)
        if check_aux:
            for _, det in self.aux_detectors:
                c_aux, ids_aux, _ = self._run_detector(det, g_small)
                d_aux = self._parse_corners(c_aux, ids_aux, fx, fy, cx_mid, h_img, scale=scale)
                if d_aux:
                    return d_aux

        return []


class SimpleSimArena:
    """
    Simulates a 360-degree arena with 3 AprilTags (T1, T2, T3) and 2 front ToF sensors
    (CH0 blocked by the arm @ 85 mm, CH1 reading straight clear distance in mm).
    """

    def __init__(self, target_tag_id: int = 3):
        self.target_tag_id = int(target_tag_id)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        b0 = math.radians(42.0)
        self.tags = {
            self.target_tag_id: {"x": 1.25 * math.cos(b0), "y": 1.25 * math.sin(b0), "yaw": b0},
            1 if self.target_tag_id != 1 else 2: {"x": 1.40 * math.cos(b0 + 2.1), "y": 1.40 * math.sin(b0 + 2.1), "yaw": wrap_angle(b0 + 2.1)},
            2 if self.target_tag_id not in (1, 2) else 4: {"x": 1.35 * math.cos(b0 - 2.2), "y": 1.35 * math.sin(b0 - 2.2), "yaw": wrap_angle(b0 - 2.2)},
        }

        self.markers = {}
        if hasattr(cv2, "aruco"):
            d36 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            for tid in self.tags:
                m = cv2.aruco.generateImageMarker(d36, int(tid), 160) if hasattr(cv2.aruco, "generateImageMarker") else cv2.aruco.drawMarker(d36, int(tid), 160)
                self.markers[int(tid)] = cv2.copyMakeBorder(m, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255)

    def step(self, vx: float, wz: float, dt: float):
        self.yaw = wrap_angle(self.yaw + float(wz) * float(dt))
        self.x += float(vx) * math.cos(self.yaw) * float(dt)
        self.y += float(vx) * math.sin(self.yaw) * float(dt)
        t = self.tags[self.target_tag_id]
        dist_m = max(0.08, math.hypot(t["x"] - self.x, t["y"] - self.y))
        tof_sub_info = [85, int(min(3500.0, dist_m * 1000.0)), 0, 0]
        return self.x, self.y, self.yaw, tof_sub_info

    def render_frame(self) -> np.ndarray:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:185, :] = (160, 165, 170)
        frame[185:, :] = (110, 115, 120)
        cv2.line(frame, (0, 185), (640, 185), (80, 85, 90), 2)

        fx, fy, cx_mid, cy_mid = 520.0, 390.0, 320.0, 175.0
        for tid, t in self.tags.items():
            dx, dy = t["x"] - self.x, t["y"] - self.y
            dist = max(0.12, math.hypot(dx, dy))
            bearing = wrap_angle(math.atan2(dy, dx) - self.yaw)
            if abs(bearing) > math.radians(42.0):
                continue
            u = int(cx_mid - fx * math.tan(bearing))
            px_h = int(np.clip((0.048 * fy) / dist, 16, 220))
            skew = wrap_angle(self.yaw - t["yaw"])
            px_w = int(max(14, px_h * max(0.65, math.cos(0.6 * skew))))
            x1, y1 = u - px_w // 2, int(cy_mid - px_h // 2)
            x2, y2 = x1 + px_w, y1 + px_h
            if 0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 360:
                m = self.markers.get(int(tid))
                if m is not None:
                    resized = cv2.resize(m, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(frame, (240, 275), (410, 359), (32, 34, 38), -1)
        return frame
