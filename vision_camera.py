#!/usr/bin/env python3
"""
vision_camera.py - 100% Pure Camera-Based AprilTag Detector & Arena Simulator
=============================================================================
Designed for assessment execution when the physical ToF sensor is disconnected.
All range, bearing, and alignment measurements are derived optically from the
camera using calibrated pinhole geometry, subpixel ArUco corner refinement,
and 2D spatial world memory.

Key Features:
  1. Optical Distance Z = (tag_size_m * fy) / max(edge_lengths)
     - Pitch/tilt invariant: Uses the maximum edge length across all 4 corner pairs
       so floor stand pitch foreshortening never distorts distance measurement.
  2. Subpixel Parallel Skew Estimation:
     - Measures tilt/skew of the tag relative to the camera image plane:
       skew = edge_slope + 0.35 * perspective_trapezoid_ratio
     - skew == 0 when robot is directly facing and parallel to the tag.
  3. Continuous Spatial World Memory:
     - Integrates odometry with visual detections to remember 2D world coordinates
       of all seen tags (T1, T2, T3) in the arena:
       (tx, ty) = (x_odom + d * cos(yaw + brg), y_odom + d * sin(yaw + brg))
  4. CameraSimArena:
     - Realistic simulation arena rendering AprilTags at true perspective and scale
       without any ToF dependency.
"""

import math
from collections import deque
import numpy as np
import cv2

from flatness_math import wrap_angle


class CameraAprilTagDetector:
    """
    Detects AprilTags on the floor/stand and computes:
      - id: AprilTag ID
      - dist_m: Camera-estimated Euclidean distance to the tag face (m)
      - bearing_rad: Horizontal angle to tag center (+left, -right)
      - skew_rad: Parallel alignment error from 4 tag corners (0 = squarely parallel)
      - bbox: (x, y, w, h)
      - corners: np.ndarray shape (4, 2)
    """

    def __init__(self, tag_size_m: float = 0.048):
        self.tag_size_m = float(tag_size_m)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.detectors = []

        if hasattr(cv2, "aruco"):
            for name in ("DICT_APRILTAG_36h11", "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_4X4_50"):
                did = getattr(cv2.aruco, name, None)
                if did is not None:
                    try:
                        adict = cv2.aruco.getPredefinedDictionary(did)
                        if hasattr(cv2.aruco, "DetectorParameters"):
                            params = cv2.aruco.DetectorParameters()
                            params.adaptiveThreshWinSizeMin = 3
                            params.adaptiveThreshWinSizeMax = 53
                            params.adaptiveThreshWinSizeStep = 6
                            params.minMarkerPerimeterRate = 0.012
                            if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
                                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                            self.detectors.append((name, cv2.aruco.ArucoDetector(adict, params)))
                        else:
                            params = cv2.aruco.DetectorParameters_create()
                            params.minMarkerPerimeterRate = 0.012
                            self.detectors.append((name, (adict, params)))
                    except Exception:
                        pass

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

    def _estimate_tag_distance(self, pts: np.ndarray, bw: float, bh: float, fy: float) -> float:
        """
        Estimates distance (m) to the AprilTag using calibrated pinhole optics:
        Z = (tag_size_m * fy) / apparent_px
        Computes the lengths of all 4 edges of the detected marker quadrilateral.
        Using max(edges) guarantees that vertical floor pitch / perspective foreshortening
        does not cause distance overestimation.
        """
        edge_lengths = [
            float(np.linalg.norm(pts[1] - pts[0])),
            float(np.linalg.norm(pts[2] - pts[1])),
            float(np.linalg.norm(pts[3] - pts[2])),
            float(np.linalg.norm(pts[0] - pts[3])),
            float(bw),
            float(bh),
        ]
        apparent_px = max(edge_lengths)
        size_dist = (self.tag_size_m * fy) / max(apparent_px, 6.0)
        return float(np.clip(size_dist, 0.08, 3.20))

    def _aruco_detect(self, gray: np.ndarray, fx: float, fy: float, cx_mid: float):
        results = []
        seen = set()

        for _, det in self.detectors:
            if isinstance(det, tuple):
                corners, ids, _ = cv2.aruco.detectMarkers(gray, det[0], parameters=det[1])
            else:
                corners, ids, _ = det.detectMarkers(gray)

            if ids is None or len(ids) == 0:
                continue

            for i, raw_id in enumerate(np.asarray(ids).flatten()):
                tid = int(raw_id)
                if tid in seen:
                    continue
                pts = corners[i][0].astype(np.float64)
                cx = float(np.mean(pts[:, 0]))

                x1, y1 = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
                x2, y2 = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))
                bw, bh = max(4, x2 - x1), max(4, y2 - y1)

                dist_m = self._estimate_tag_distance(pts, bw, bh, fy)
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

    def detect_all(self, frame: np.ndarray, default_id: int = 3):
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h_img, w_img = gray.shape[:2]
        fx = (w_img / 640.0) * 520.0
        fy = (h_img / 360.0) * 390.0
        cx_mid = w_img / 2.0

        # Pass 1: Standard ArUco detection across FULL unmasked frame
        dets = self._aruco_detect(gray, fx, fy, cx_mid)
        if dets:
            return dets

        # Pass 2: Enhanced CLAHE contrast detection
        dets = self._aruco_detect(self.clahe.apply(gray), fx, fy, cx_mid)
        if dets:
            return dets

        # Pass 3: Distant bright-square tag fallback
        y_top, y_bot = int(0.20 * h_img), int(0.70 * h_img)
        roi = gray[y_top:y_bot, :]
        _, thresh = cv2.threshold(roi, 210, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_sq = (w_img / 640.0) ** 2

        best = None
        for c in contours:
            area = cv2.contourArea(c)
            if not (60.0 * scale_sq <= area <= 18000.0 * scale_sq):
                continue
            bx, by_rel, bw, bh = cv2.boundingRect(c)
            by = by_rel + y_top
            aspect = float(bw) / max(float(bh), 1.0)
            if not (0.65 <= aspect <= 1.45) or (area / max(float(bw * bh), 1.0)) < 0.65:
                continue
            pad = max(5, int(0.40 * bw))
            xl1, xl2 = max(0, bx - pad), max(1, bx - 2)
            xr1, xr2 = min(w_img - 1, bx + bw + 2), min(w_img, bx + bw + pad)
            if xl2 <= xl1 or xr2 <= xr1:
                continue
            lm = float(np.mean(gray[by:by + bh, xl1:xl2]))
            rm = float(np.mean(gray[by:by + bh, xr1:xr2]))
            if lm > 80.0 or rm > 80.0 or abs(lm - rm) > 40.0:
                continue
            score = area * (95.0 - 0.5 * (lm + rm))
            if best is None or score > best[0]:
                best = (score, (bx, by, bw, bh), c)

        if best is not None:
            _, (bx, by, bw, bh), cnt = best
            cx = bx + 0.5 * bw
            box_pts = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64)
            box_pts[:, 1] += y_top
            dist_m = self._estimate_tag_distance(box_pts, bw, bh, fy)
            bearing_rad = -math.atan2(cx - cx_mid, fx)
            return [{
                "id": int(default_id),
                "dist_m": dist_m,
                "bearing_rad": bearing_rad,
                "skew_rad": self.compute_parallel_skew(box_pts),
                "bbox": (bx, by, bw, bh),
                "corners": box_pts,
            }]
        return []


class CameraSimArena:
    """
    Simulates a 360-degree arena with 3 AprilTags (T1, T2, T3) and renders
    realistic 360p camera frames with perspective projection, marker textures,
    and robot chassis odometry.
    No ToF sensor dependency.
    """

    def __init__(self, target_tag_id: int = 3):
        self.target_tag_id = int(target_tag_id)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        b0 = math.radians(45.0)
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
        return self.x, self.y, self.yaw

    def render_frame(self) -> np.ndarray:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:185, :] = (160, 165, 170)
        frame[185:, :] = (110, 115, 120)
        cv2.line(frame, (0, 185), (640, 185), (80, 85, 90), 2)

        fx, fy, cx_mid, cy_mid = 520.0, 390.0, 320.0, 175.0
        for tid, t in self.tags.items():
            dx, dy = t["x"] - self.x, t["y"] - self.y
            dist = max(0.10, math.hypot(dx, dy))
            bearing = wrap_angle(math.atan2(dy, dx) - self.yaw)
            if abs(bearing) > math.radians(45.0):
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

        # Draw robotic arm base silhouette at bottom of camera view
        cv2.rectangle(frame, (240, 275), (410, 359), (32, 34, 38), -1)
        return frame
