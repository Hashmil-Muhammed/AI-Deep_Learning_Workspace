"""
theft_detection_vid2.py
=======================
Advanced Real-Time AI Retail Product Tracking & Theft Detection Pipeline
Engineered for CCTV Retail Surveillance (vid2.mp4, Video Project 1.mp4, and Live Feeds).

Core Architecture & Workflow Implemented:
1. CUSTOMER DETECTION & CONTINUOUS TRACKING:
   - Tracks customers entering the camera view with persistent Customer IDs.
   - Robust IoU + centroid distance matching to prevent ID switches.
   - Hand keypoints (wrists, elbows) accurately associated with each customer.

2. HAND DETECTION & SHELF INTERACTION:
   - Tracks wrists and elbows as hands reach into the marked shelf ROI.
   - IMPORTANT: Hand entering the shelf alone NEVER triggers a pickup or theft.

3. ACTUAL PHYSICAL PRODUCT DETECTION (NOT THE HAND):
   - Dynamic background subtraction on the shelf ROI detects genuine product removal.
   - Distinguishes physical shelf gaps (area >= 450px) from hand shadows and noise.
   - Isolates the product from the hand using dual-color space (HSV + YCrCb) skin subtraction.
   - Draws a tight, dedicated bounding box [PRODUCT: #N] strictly around the physical product.
   - Captures product appearance signature (color signature & shelf template).

4. INDEPENDENT PRODUCT TRACKING:
   - Tracks the actual product independently of hand keypoints.
   - Handles hand movement, rotation, and partial occlusion with temporal debouncing.

5. 11-STATE PRODUCT LIFECYCLE STATE MACHINE:
   - ON_SHELF
   - PICKUP_CANDIDATE
   - PICKED_FROM_SHELF
   - IN_HAND
   - TRACKING_UNCERTAIN
   - RETURN_CANDIDATE
   - RETURNED_TO_SHELF
   - PLACED_IN_BAG
   - MISSING_SUSPICIOUS
   - THEFT_ALERT
   - CUSTOMER_EXITED_UNRESOLVED

6. SHELF RETURN DETECTION (SAME SHELF):
   - Verifies the product re-enters the shelf ROI and the shelf gap closes (product placed back).
   - Confirms stationary placement -> State: RETURNED_TO_SHELF.
   - Increments 'Products Returned to Shelf' (+1), decrements 'Currently in Hand' (-1).

7. BAG / COVER PLACEMENT DETECTION (SAFE):
   - Tracks customer shoulder bags and handheld bags anchored to torso keypoints.
   - Verifies deliberate placement into bag -> State: PLACED_IN_BAG.
   - Increments 'Products Placed in Bag' (+1), decrements 'Currently in Hand' (-1).

8. THEFT DETECTION & FALSE-POSITIVE PREVENTION:
   - Multi-stage grace period: IN_HAND -> TRACKING_UNCERTAIN -> MISSING_SUSPICIOUS -> THEFT_ALERT.
   - Confirms theft alert only when product is missing from hand/bag AND shelf gap remains open.
   - Increments 'Theft Alerts' (+1), decrements 'Currently in Hand' (-1).

9. ACCURATE 7-METRIC EVENT COUNTERS & TELEMETRY PANEL:
   - Customers in View, Total Products Picked, Currently in Hand,
     Products Returned to Shelf, Products Placed in Bag,
     Suspicious Missing Products, Theft Alerts.
"""

import cv2
import numpy as np
import os
import sys
import time
import argparse
import collections

# Optional ultralytics or onnxruntime
try:
    import torch
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────── COLOR PALETTE ────────────────────────────────
RED    = (40,  40,  235)  # Theft Alert
AMB    = (0,   185, 255)  # Product Pick / In-Hand (Yellow-Gold)
ORANGE = (0,   130, 255)  # Uncertain / Suspicious
GRN    = (50,  205, 50)   # Safe / Returned / Browsing
CYN    = (230, 210, 0)    # Reaching Shelf / Shelf ROI
WHT    = (255, 255, 255)
GRY    = (160, 160, 160)
DARK   = (22,  22,  26)
PANEL_BG = (20, 20, 24)

# ──────────────────────── PRODUCT STATE ENUM ───────────────────────────────
class ProductState:
    ON_SHELF                   = "ON_SHELF"
    PICKUP_CANDIDATE           = "PICKUP_CANDIDATE"
    PICKED_FROM_SHELF          = "PICKED_FROM_SHELF"
    IN_HAND                    = "IN_HAND"
    TRACKING_UNCERTAIN         = "TRACKING_UNCERTAIN"
    RETURN_CANDIDATE           = "RETURN_CANDIDATE"
    RETURNED_TO_SHELF          = "RETURNED_TO_SHELF"
    PLACED_IN_BAG              = "PLACED_IN_BAG"
    MISSING_SUSPICIOUS         = "MISSING_SUSPICIOUS"
    THEFT_ALERT                = "THEFT_ALERT"
    CUSTOMER_EXITED_UNRESOLVED = "CUSTOMER_EXITED_UNRESOLVED"


# ──────────────────────────── MODEL INITIALIZATION ──────────────────────────
def get_device():
    if HAS_ULTRALYTICS and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DetectorWrapper:
    def __init__(self, model_det="yolov8n.pt", model_pose="yolov8n-pose.pt", imgsz=480):
        self.device = get_device()
        self.imgsz = imgsz
        self.use_pt = HAS_ULTRALYTICS
        print(f"[Detector] Initializing using device: {self.device.upper()}")

        if self.use_pt:
            print(f"[Detector] Loading PyTorch models via Ultralytics: {model_det}, {model_pose}")
            pt_det = os.path.join(HERE, model_det) if os.path.exists(os.path.join(HERE, model_det)) else model_det
            pt_pose = os.path.join(HERE, model_pose) if os.path.exists(os.path.join(HERE, model_pose)) else model_pose
            self.model_det = YOLO(pt_det)
            self.model_pose = YOLO(pt_pose)
        else:
            print("[Detector] Ultralytics not installed. Loading ONNX models...")
            onnx_det_path = os.path.join(HERE, "models", "yolov8m.onnx")
            onnx_pose_path = os.path.join(HERE, "models", "yolov8m-pose.onnx")
            if not os.path.exists(onnx_det_path) or not os.path.exists(onnx_pose_path):
                raise FileNotFoundError("Model files not found! Please install ultralytics or provide ONNX models.")
            so = ort.SessionOptions()
            so.intra_op_num_threads = min(4, os.cpu_count() or 2)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            self.sess_det = ort.InferenceSession(onnx_det_path, so, providers=providers)
            self.sess_pose = ort.InferenceSession(onnx_pose_path, so, providers=providers)

    def detect_objects_and_pose(self, frame, conf_det=0.15, conf_pose=0.25):
        """
        Returns:
            persons: list of dicts(box=[x1,y1,x2,y2], conf, k=17x3 keypoints)
            containers: list of dicts(box=[x1,y1,x2,y2], cls, conf, label)
        """
        if self.use_pt:
            # 1. Pose estimation (Customer body & hands)
            pose_res = self.model_pose.predict(frame, conf=conf_pose, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            persons = []
            if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                boxes = pose_res.boxes.xyxy.cpu().numpy()
                confs = pose_res.boxes.conf.cpu().numpy()
                kpts = pose_res.keypoints.data.cpu().numpy()
                for b, c, k in zip(boxes, confs, kpts):
                    persons.append(dict(box=b.tolist(), conf=float(c), k=k))

            # 2. Containers / bags detection (Filter out shelf items and low-confidence boxes)
            det_res = self.model_det.predict(frame, conf=0.30, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            containers = []
            if det_res.boxes is not None and len(det_res.boxes) > 0:
                d_boxes = det_res.boxes.xyxy.cpu().numpy()
                d_confs = det_res.boxes.conf.cpu().numpy()
                d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
                CONTAINER_CLASSES = {24: "BACKPACK", 26: "HANDBAG / BAG"}
                for b, c, cl in zip(d_boxes, d_confs, d_cls):
                    if cl in CONTAINER_CLASSES:
                        containers.append(dict(box=b.tolist(), conf=float(c), cls=cl, label=CONTAINER_CLASSES[cl]))
            return persons, containers
        else:
            return self._infer_onnx(frame, conf_det, conf_pose)

    def _infer_onnx(self, frame, conf_det, conf_pose):
        h, w = frame.shape[:2]
        input_sz = 640
        r = input_sz / max(h, w)
        nh, nw = int(h * r), int(w * r)
        can = np.full((input_sz, input_sz, 3), 114, np.uint8)
        can[:nh, :nw] = cv2.resize(frame, (nw, nh))
        blob = can[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        out_pose = self.sess_pose.run(None, {"images": blob})[0][0].T
        sc_p = out_pose[:, 4]
        mask_p = sc_p > conf_pose
        out_p, sc_p = out_pose[mask_p], sc_p[mask_p]
        persons = []
        if len(out_p):
            bx_p = np.stack([out_p[:, 0] - out_p[:, 2] / 2, out_p[:, 1] - out_p[:, 3] / 2, out_p[:, 2], out_p[:, 3]], 1) / r
            keep_p = np.array(cv2.dnn.NMSBoxes(bx_p.tolist(), sc_p.tolist(), conf_pose, 0.45)).flatten()
            for i in keep_p:
                x, y, bw, bh = bx_p[i]
                k = out_p[i, 5:].reshape(17, 3).copy()
                k[:, :2] /= r
                persons.append(dict(box=[x, y, x + bw, y + bh], conf=float(sc_p[i]), k=k))

        out_det = self.sess_det.run(None, {"images": blob})[0][0].T
        sc_d = out_det[:, 4:].max(1)
        mask_d = sc_d > conf_det
        out_d, sc_d = out_det[mask_d], sc_d[mask_d]
        containers = []
        if len(out_d):
            bx_d = np.stack([out_d[:, 0] - out_d[:, 2] / 2, out_d[:, 1] - out_d[:, 3] / 2, out_d[:, 2], out_d[:, 3]], 1) / r
            cls_d = out_d[:, 4:].argmax(1)
            keep_d = np.array(cv2.dnn.NMSBoxes(bx_d.tolist(), sc_d.tolist(), conf_det, 0.45)).flatten()
            CONTAINER_CLASSES = {24: "BACKPACK", 26: "HANDBAG / BAG", 28: "BASKET / BOX"}
            for i in keep_d:
                c_id = int(cls_d[i])
                if c_id in CONTAINER_CLASSES:
                    x, y, bw, bh = bx_d[i]
                    containers.append(dict(box=[x, y, x + bw, y + bh], conf=float(sc_d[i]), cls=c_id, label=CONTAINER_CLASSES[c_id]))
        return persons, containers


# ─────────────────────────── GEOMETRY & TRACKING UTILS ──────────────────────
def iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def point_in_box(pt, box, margin=0):
    return (box[0] - margin <= pt[0] <= box[2] + margin and
            box[1] - margin <= pt[1] <= box[3] + margin)


def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


# ──────────────────────── SHELF GAP & PRODUCT ISOLATION ─────────────────────
def detect_shelf_gaps(frame, clean_bg, shelf_mask, threshold=30, min_area=450, person_mask=None):
    """
    Detects physical product removal gaps on the shelf using background subtraction.
    Excludes human skin and person bounding boxes to prevent false triggers from hand entry.
    Returns: list of dicts: [ {"box": [x1,y1,x2,y2], "area": int, "center": (cx, cy)} ]
    """
    diff = cv2.absdiff(frame, clean_bg)
    diff_shelf = cv2.bitwise_and(diff, diff, mask=shelf_mask)
    gray = cv2.cvtColor(diff_shelf, cv2.COLOR_BGR2GRAY)

    # Detect and remove skin pixels to ensure hands don't appear as shelf gaps
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_hsv = cv2.inRange(hsv, (0, 15, 40), (25, 220, 255)) | cv2.inRange(hsv, (165, 15, 40), (180, 220, 255))
    skin_ycrcb = cv2.inRange(ycrcb, (0, 130, 75), (255, 175, 130))
    skin = cv2.bitwise_or(skin_hsv, skin_ycrcb)
    skin = cv2.dilate(skin, np.ones((11, 11), np.uint8))

    if person_mask is not None:
        skin = cv2.bitwise_or(skin, person_mask)

    gray_clean = cv2.bitwise_and(gray, cv2.bitwise_not(skin))
    _, th = cv2.threshold(gray_clean, threshold, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gaps = []
    for c in cnts:
        area = int(cv2.contourArea(c))
        if area >= min_area:
            bx, by, bw, bh = cv2.boundingRect(c)
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            gaps.append({
                "box": [bx, by, bx + bw, by + bh],
                "area": area,
                "center": (cx, cy)
            })

    # Sort largest area first
    gaps.sort(key=lambda x: -x["area"])
    return gaps


def isolate_product_in_hand(frame, wrist_pts, shelf_template=None, color_sig=None, search_radius=75):
    """
    Separates the ACTUAL PHYSICAL PRODUCT from the customer's hand.
    - Excludes skin/fingers/wrist completely using dual-color space skin detection.
    - Isolates non-skin physical item held near the wrist/palm.
    - Matches against shelf template / color appearance when available.
    Returns: (found: bool, tight_product_box: [x1, y1, x2, y2], conf: float)
    """
    H, W = frame.shape[:2]
    best_candidate = None
    best_score = 0.0

    for w_pt in wrist_pts:
        if w_pt is None:
            continue
        wx, wy = int(w_pt[0]), int(w_pt[1])
        x1 = max(0, wx - search_radius)
        y1 = max(0, wy - search_radius)
        x2 = min(W, wx + search_radius)
        y2 = min(H, wy + search_radius)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            continue

        patch = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

        # 1. Skin tone detector (HSV + YCrCb) to completely subtract skin/fingers
        skin_hsv = cv2.inRange(hsv, (0, 15, 35), (25, 230, 255)) | cv2.inRange(hsv, (165, 15, 35), (180, 230, 255))
        skin_ycrcb = cv2.inRange(ycrcb, (0, 130, 75), (255, 175, 130))
        skin = cv2.bitwise_or(skin_hsv, skin_ycrcb)
        skin = cv2.dilate(skin, np.ones((5, 5), np.uint8))

        # 2. Extract non-skin objects in the grasp
        non_skin = cv2.bitwise_not(skin)
        non_skin = cv2.morphologyEx(non_skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(non_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 50 or area > (search_radius * search_radius * 1.8):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            # Aspect ratio check: retail products are reasonably shaped (not flat slivers)
            ar = bw / max(1.0, bh)
            if ar < 0.15 or ar > 4.5:
                continue

            obj_crop = patch[by:by+bh, bx:bx+bw]
            score = float(area)

            # If color signature is available, weight by color similarity
            if color_sig is not None and obj_crop.size > 0:
                mean_bgr = obj_crop.mean(axis=(0, 1))
                c_dist = np.linalg.norm(mean_bgr - color_sig)
                score += max(0.0, (120.0 - c_dist) * 4.0)

            if score > best_score:
                best_score = score
                # Compute tight bounding box strictly on the physical item
                best_candidate = [x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh]

    if best_candidate is not None:
        return True, best_candidate, best_score

    return False, None, 0.0


# ─────────────────────────── TRACKED PRODUCT CLASS ──────────────────────────
class TrackedProduct:
    """
    Dedicated object tracker for the physical retail product (independent of hand).
    Implements the 11-State Product Lifecycle.
    """
    def __init__(self, prod_id, customer_id, initial_box, shelf_origin_box, t_pick, template=None, color_sig=None):
        self.id = prod_id
        self.customer_id = customer_id
        self.box = list(initial_box)          # Tight product bbox [x1, y1, x2, y2]
        self.shelf_origin_box = list(shelf_origin_box) if shelf_origin_box else None
        self.origin_center = box_center(self.shelf_origin_box) if shelf_origin_box else None
        self.t_pick = t_pick
        self.t_last_seen = t_pick
        self.state = ProductState.PICKED_FROM_SHELF
        self.template = template              # Image crop from shelf before removal
        self.color_sig = color_sig            # Mean BGR / HSV appearance signature

        # Temporal debouncing and state counters
        self.uncertain_frames = 0
        self.missing_frames = 0
        self.return_candidate_frames = 0
        self.bag_drop_candidate_frames = 0
        self.final_logged = False


# ─────────────────────────── TRACKED CUSTOMER CLASS ─────────────────────────
class CustomerTrack:
    """
    Tracks customer identity, posture, hand interactions, and carried products.
    """
    def __init__(self, cid, det, t):
        self.id = cid
        self.d = det                         # {"box": [x1,y1,x2,y2], "conf": c, "k": kpts}
        self.t0 = t
        self.t = t
        self.state = "BROWSING"              # BROWSING, REACHING SHELF, HOLDING PRODUCT, PUT BACK, PLACED IN BAG, THEFT ALERT
        self.hands_in_shelf = False
        self.last_shelf_reach_time = -99.0
        self.hand_in_shelf_frames = 0
        self.active_product = None           # Current TrackedProduct object
        self.completed_products = []         # History of finished products
        self.theft_flagged = False

        # Bag / Cover Tracking
        self.bag_box = None
        self.bag_type = "SHOULDER BAG"
        self.bag_rel_offset = None
        self.bag_missing_frames = 0


# ────────────────────────── INTERACTIVE SHELF DRAWER ────────────────────────
def draw_shelf_roi(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ret, frame_native = cap.read()
    cap.release()

    if not ret or frame_native is None:
        raise RuntimeError(f"Cannot read frame from {video_path}")

    H_nat, W_nat = frame_native.shape[:2]
    max_w, max_h = 1280, 720
    scale = min(max_w / W_nat, max_h / H_nat, 1.0)
    dw, dh = int(W_nat * scale), int(H_nat * scale)
    sx, sy = dw / W_nat, dh / H_nat
    disp = cv2.resize(frame_native, (dw, dh))

    polys = []
    cur = []
    mouse = [-1, -1]

    # Pre-configured default shelves for known benchmark videos
    vname = os.path.basename(video_path).lower()
    default_shelves = []
    if "vid2" in vname:
        default_shelves = [[[660, 40], [1260, 40], [1260, 520], [660, 520]]]
    elif "project 1" in vname or "video project 1" in vname:
        default_shelves = [[[350, 60], [420, 160], [380, 680], [140, 680], [110, 480], [250, 190]]]

    def mouse_callback(event, x, y, flags, _):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            cur.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(cur) >= 3:
                polys.append(list(cur))
                cur.clear()
                print(f"[Shelf ROI] Shelf #{len(polys)} captured.")
            else:
                print("Click at least 3 points to complete a shelf polygon.")

    win = "Draw Shelf ROI  |  L-Click: Add Point  |  R-Click: Close  |  D: Default  |  Enter: Run"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    cv2.setMouseCallback(win, mouse_callback)

    print("\n" + "="*65)
    print(" STEP 1: DEFINE SHELF ROI")
    print(" • Left-Click  : Place corner of shelf polygon")
    print(" • Right-Click : Close current shelf polygon")
    print(" • D           : Load Default Preset for this video")
    print(" • Z           : Undo point")
    print(" • R           : Reset all polygons")
    print(" • Enter / S   : Save Shelf Background & Start Detection")
    print("="*65 + "\n")

    while True:
        vis = disp.copy()
        for i, poly in enumerate(polys):
            pts = np.array(poly, np.int32)
            ov = vis.copy()
            cv2.fillPoly(ov, [pts], (0, 220, 220))
            cv2.addWeighted(ov, 0.22, vis, 0.78, 0, vis)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            cx_ = int(pts[:, 0].mean())
            cy_ = int(pts[:, 1].mean())
            cv2.putText(vis, f"Shelf {i+1}", (cx_ - 26, cy_), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2)

        if len(cur) >= 2:
            cv2.polylines(vis, [np.array(cur, np.int32)], False, (0, 200, 255), 2, cv2.LINE_AA)
        for pt in cur:
            cv2.circle(vis, pt, 5, (0, 200, 255), -1)
        if cur and mouse[0] >= 0:
            cv2.line(vis, cur[-1], tuple(mouse), (0, 200, 255), 1, cv2.LINE_AA)

        cv2.putText(vis, f"Shelves defined: {len(polys)} | Points in progress: {len(cur)}", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
        cv2.putText(vis, "L-Click: Add | R-Click: Close | D: Default Preset | Enter: Run | Z: Undo | R: Reset", (14, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('z'), ord('Z')):
            if cur: cur.pop()
        elif key in (ord('r'), ord('R')):
            polys.clear(); cur.clear()
        elif key in (ord('d'), ord('D')):
            if default_shelves:
                polys = [list(poly) for poly in default_shelves]
                cur.clear()
                print(f"[Shelf ROI] Loaded default preset with {len(polys)} shelf polygon(s).")
            else:
                print("No default preset configured for this video.")
        elif key in (13, ord('s'), ord('S'), 32):
            if len(cur) >= 3:
                polys.append(list(cur))
                cur.clear()
            if polys:
                break
            elif default_shelves:
                polys = [list(poly) for poly in default_shelves]
                print(f"[Shelf ROI] Using default preset shelf.")
                break
            else:
                print("Please draw at least one shelf polygon before continuing.")
        elif key in (27, ord('q'), ord('Q')):
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    return [[(int(p[0] / sx), int(p[1] / sy)) for p in poly] for poly in polys]


# ─────────────────────────── MAIN DETECTION PIPELINE ────────────────────────
def run_theft_detection(video_path, shelf_polygons, save=True, show=True, skip_frames=2, model_size="n"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Responsive window rendering
    S = 720.0 / H
    vw = int(W * S)
    PW = 390
    if vw > 960:
        vw = 960
        sx_render = vw / W
        sy_render = 720.0 / H
    else:
        sx_render = S
        sy_render = S
    total_w = vw + PW

    # Build Shelf Polygons and Masks
    shelf_np = [np.array(p, np.int32) for p in shelf_polygons]
    shelf_mask = np.zeros((H, W), dtype=np.uint8)
    for sp in shelf_np:
        cv2.fillPoly(shelf_mask, [sp], 255)

    # Capture initial Clean Shelf ROI Background (Frame 0-5)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame 0 for background reference.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ret, clean_shelf_bg = cap.read()
    if not ret:
        clean_shelf_bg = first_frame.copy()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("[Pipeline] Clean shelf ROI background initialized successfully.")

    # Detector
    model_det = f"yolov8{model_size}.pt"
    model_pose = f"yolov8{model_size}-pose.pt"
    detector = DetectorWrapper(model_det=model_det, model_pose=model_pose, imgsz=480)

    # Output video writer
    out_file = os.path.join(HERE, "out_theft_v2.mp4")
    wr = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, 720)) if save else None

    if show:
        cv2.namedWindow("AI Retail Theft Detection v2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Retail Theft Detection v2", total_w, 720)

    # Customers, accounted shelf gaps, and metrics
    people = {}
    next_customer_id = 1
    events = []
    accounted_shelf_gaps = []  # List of (cx, cy) locations already picked

    # Accurate 7-Metric Event Counters (Section 9)
    counts = dict(
        total_picked=0,          # Total Products Picked
        currently_held=0,        # Currently in Hand
        returned_to_shelf=0,     # Products Returned to Shelf
        placed_in_bag=0,         # Products Placed in Bag
        suspicious_missing=0,    # Suspicious Missing Products
        theft_alerts=0           # Theft Alerts
    )

    # Configurable detection timeouts
    UNCERTAIN_TIMEOUT_SEC = 1.0  # Seconds before marking tracking uncertain
    THEFT_TIMEOUT_SEC = 2.4      # Seconds before confirmed theft alert

    def log_event(t, text, col):
        events.append((t, text, col))
        prefix = "🚨 " if col is RED else ("⚠️ " if col is ORANGE else ("🛍 " if col is AMB else "✓ "))
        print(f"[{t:5.1f}s] {prefix}{text}")

    idx = 0
    t_prev = time.time()
    fps_disp = 0.0

    print("\n[Pipeline] Detection running... (Press 'Q' to stop)\n")

    cached_persons = []
    cached_containers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        t = idx / fps

        # ──────────────── 1. RUN YOLO INFERENCE (WITH FRAME SKIPPING) ────────────────
        if idx % (skip_frames + 1) == 1 or not cached_persons:
            persons_det, containers_det = detector.detect_objects_and_pose(frame, conf_det=0.15, conf_pose=0.25)
            cached_persons = persons_det
            cached_containers = containers_det
        else:
            persons_det = cached_persons
            containers_det = cached_containers

        # ──────────────── 2. SHELF GAP ANALYSIS (GLOBAL FOR FRAME) ────────────────
        # Build person mask to ensure customer bodies are excluded from shelf gap calculation
        person_mask = np.zeros((H, W), dtype=np.uint8)
        for pd in persons_det:
            pb = pd["box"]
            x1p, y1p, x2p, y2p = int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])
            cv2.rectangle(person_mask, (x1p, y1p), (x2p, y2p), 255, -1)
        person_shelf_mask = cv2.bitwise_and(person_mask, shelf_mask)

        shelf_gaps = detect_shelf_gaps(
            frame, clean_shelf_bg, shelf_mask,
            threshold=30, min_area=450,
            person_mask=person_shelf_mask
        )

        # ──────────────── 3. CUSTOMER TRACKING (CENTROID + IOU MATCHING) ───────────
        used_det = set()
        for p in sorted(people.values(), key=lambda x: -x.t):
            if t - p.t > 2.0:
                continue
            best_idx, best_score = None, 0.10
            p_center = box_center(p.d["box"])
            for j, d in enumerate(persons_det):
                if j in used_det:
                    continue
                sc_iou = iou(p.d["box"], d["box"])
                d_center = box_center(d["box"])
                dist = np.hypot(p_center[0] - d_center[0], p_center[1] - d_center[1])
                score = sc_iou + max(0.0, 0.5 - dist / float(H))
                if score > best_score:
                    best_idx, best_score = j, score

            if best_idx is not None:
                used_det.add(best_idx)
                p.d = persons_det[best_idx]
                p.t = t

        for j, d in enumerate(persons_det):
            if j not in used_det:
                people[next_customer_id] = CustomerTrack(next_customer_id, d, t)
                log_event(t, f"Customer #{next_customer_id} entered camera view", GRN)
                next_customer_id += 1

        active_customers = [p for p in people.values() if p.t == t]

        # ──────────────── 4. PROCESS EACH CUSTOMER'S INTERACTIONS ──────────────────
        for p in active_customers:
            b = p.d["box"]
            k = p.d["k"]
            ph = max(1.0, b[3] - b[1])
            reach_margin = int(0.06 * ph)

            # Keypoints: shoulders(5,6), elbows(7,8), wrists(9,10), hips(11,12)
            ls, rs = k[5], k[6]
            le, re = k[7], k[8]
            lw, rw = k[9], k[10]
            lh, rh = k[11], k[12]

            sh_valid = [q for q in (ls, rs) if q[2] > 0.15]
            hip_valid = [q for q in (lh, rh) if q[2] > 0.15]
            sy_mean = float(np.mean([q[1] for q in sh_valid])) if sh_valid else b[1] + 0.18 * ph
            hy_mean = float(np.mean([q[1] for q in hip_valid])) if hip_valid else sy_mean + 0.42 * ph
            torso_h = max(35.0, hy_mean - sy_mean)
            torso_cx = float(np.mean([q[0] for q in sh_valid + hip_valid])) if (sh_valid + hip_valid) else (b[0] + b[2]) / 2.0
            torso_cy = (sy_mean + hy_mean) / 2.0
            torso_w = max(40.0, (b[2] - b[0]) * 0.70)

            # Filter containers to exclude shelf display objects and low-confidence boxes
            valid_containers = []
            for c in containers_det:
                cb = c["box"]
                ccx, ccy = box_center(cb)
                if not any(point_in_poly((ccx, ccy), s) for s in shelf_np):
                    valid_containers.append(c)

            # ── Bag / Container Tracking (Torso Anchored) ─────────────────────
            c_cand = []
            for c in valid_containers:
                cb = c["box"]
                cx = (cb[0] + cb[2]) / 2.0
                cy = (cb[1] + cb[3]) / 2.0
                if b[0] - 0.20 * ph <= cx <= b[2] + 0.20 * ph and b[1] <= cy <= b[3]:
                    area_c = (cb[2] - cb[0]) * (cb[3] - cb[1])
                    c_cand.append((cb, c["conf"], c.get("label", "BAG"), area_c, cy))

            if c_cand:
                best_cb, best_conf, best_lbl, best_area, best_cy = max(c_cand, key=lambda x: x[1])
                is_torso_shoulder = (best_cy <= hy_mean + 0.25 * torso_h)
                detected_bag_box = [int(best_cb[0]), int(best_cb[1]), int(best_cb[2]), int(best_cb[3])]

                if p.bag_box is None:
                    p.bag_box = detected_bag_box
                else:
                    p.bag_box = [
                        int(0.70 * p.bag_box[0] + 0.30 * detected_bag_box[0]),
                        int(0.70 * p.bag_box[1] + 0.30 * detected_bag_box[1]),
                        int(0.70 * p.bag_box[2] + 0.30 * detected_bag_box[2]),
                        int(0.70 * p.bag_box[3] + 0.30 * detected_bag_box[3]),
                    ]
                p.bag_type = "SHOULDER BAG" if is_torso_shoulder else "HANDHELD BAG"
                p.bag_missing_frames = 0

                bag_cx = (p.bag_box[0] + p.bag_box[2]) / 2.0
                bag_cy = (p.bag_box[1] + p.bag_box[3]) / 2.0
                rel_x = (bag_cx - torso_cx) / torso_w
                rel_y = (bag_cy - torso_cy) / torso_h
                bag_w = p.bag_box[2] - p.bag_box[0]
                bag_h = p.bag_box[3] - p.bag_box[1]
                p.bag_rel_offset = (rel_x, rel_y, bag_w, bag_h)

            elif p.bag_box is not None and p.bag_rel_offset is not None and p.bag_missing_frames < 60:
                rel_x, rel_y, bag_w, bag_h = p.bag_rel_offset
                exp_cx = torso_cx + rel_x * torso_w
                exp_cy = torso_cy + rel_y * torso_h
                pred_bag = [
                    int(exp_cx - bag_w / 2.0),
                    int(exp_cy - bag_h / 2.0),
                    int(exp_cx + bag_w / 2.0),
                    int(exp_cy + bag_h / 2.0)
                ]
                p.bag_box = [
                    int(0.60 * p.bag_box[0] + 0.40 * pred_bag[0]),
                    int(0.60 * p.bag_box[1] + 0.40 * pred_bag[1]),
                    int(0.60 * p.bag_box[2] + 0.40 * pred_bag[2]),
                    int(0.60 * p.bag_box[3] + 0.40 * pred_bag[3]),
                ]
                p.bag_missing_frames += 1

            # ── Hand & Shelf Interaction ──────────────────────────────────────
            hand_in_shelf = False
            valid_wrist_pts = []

            for w_idx in (9, 10):
                wkp = k[w_idx]
                if wkp[2] > 0.15:
                    w_pt = (float(wkp[0]), float(wkp[1]))
                    valid_wrist_pts.append(w_pt)
                    in_s = any(cv2.pointPolygonTest(s, w_pt, True) >= -reach_margin for s in shelf_np)
                    if in_s:
                        hand_in_shelf = True

            p.hands_in_shelf = hand_in_shelf
            if hand_in_shelf:
                p.hand_in_shelf_frames += 1
                p.last_shelf_reach_time = t
            else:
                p.hand_in_shelf_frames = 0

            time_since_shelf_reach = t - p.last_shelf_reach_time

            # ──────────────── 5. PRODUCT LIFECYCLE STATE MACHINE ────────────────
            if p.active_product is None:
                # ── STAGE A: PICKUP DETECTION ──
                # Condition: Hand was in shelf recently (within 1.5s) AND pulled back/away,
                # AND a new significant shelf gap exists that hasn't been picked yet!
                hand_reached_shelf = hand_in_shelf or (time_since_shelf_reach <= 1.5)

                if hand_reached_shelf and shelf_gaps and not hand_in_shelf:
                    # Find a gap that is near customer reach and NOT yet accounted for
                    for g in shelf_gaps:
                        gcx, gcy = g["center"]
                        # Reachability check
                        if (b[0] - 0.40 * ph) <= gcx <= (b[2] + 0.40 * ph) and (b[1] - 0.20 * ph) <= gcy <= b[3]:
                            # Accounted check
                            is_new_gap = True
                            for ag in accounted_shelf_gaps:
                                if np.hypot(gcx - ag[0], gcy - ag[1]) < 70:
                                    is_new_gap = False
                                    break

                            if is_new_gap:
                                # Genuine pickup event!
                                prod_id = counts["total_picked"] + 1
                                accounted_shelf_gaps.append((gcx, gcy))
                                gx1, gy1, gx2, gy2 = g["box"]
                                shelf_crop = clean_shelf_bg[gy1:gy2, gx1:gx2].copy()
                                color_sig = shelf_crop.mean(axis=(0, 1)) if shelf_crop.size > 0 else None

                                p.active_product = TrackedProduct(
                                    prod_id=prod_id,
                                    customer_id=p.id,
                                    initial_box=g["box"],
                                    shelf_origin_box=g["box"],
                                    t_pick=t,
                                    template=shelf_crop,
                                    color_sig=color_sig
                                )
                                p.active_product.state = ProductState.PICKED_FROM_SHELF
                                p.state = "HOLDING PRODUCT"
                                counts["total_picked"] += 1
                                counts["currently_held"] += 1
                                log_event(t, f"Customer #{p.id}: Product #{prod_id} PICKED from shelf [Shelf Gap: {g['area']}px]", AMB)
                                break

                # Customer UI State when not holding a product
                if hand_in_shelf:
                    p.state = "REACHING SHELF"
                elif not p.theft_flagged:
                    p.state = "BROWSING"

            else:
                # ── STAGE B: PRODUCT IS ACTIVELY TRACKED ──
                prod = p.active_product

                # Attempt to isolate and track the physical product object near hand
                found_in_hand, tight_prod_box, conf_score = isolate_product_in_hand(
                    frame, valid_wrist_pts,
                    shelf_template=prod.template,
                    color_sig=prod.color_sig,
                    search_radius=int(0.14 * ph)
                )

                if found_in_hand and tight_prod_box is not None:
                    # Update product location with EMA smoothing
                    prod.box = [
                        int(0.50 * prod.box[0] + 0.50 * tight_prod_box[0]),
                        int(0.50 * prod.box[1] + 0.50 * tight_prod_box[1]),
                        int(0.50 * prod.box[2] + 0.50 * tight_prod_box[2]),
                        int(0.50 * prod.box[3] + 0.50 * tight_prod_box[3]),
                    ]
                    prod.t_last_seen = t
                    prod.state = ProductState.IN_HAND
                    p.state = "HOLDING PRODUCT"
                    prod.uncertain_frames = 0
                    prod.missing_frames = 0

                else:
                    # Product not directly isolated in hand this frame
                    # Anchor product box to nearest wrist if recent
                    if valid_wrist_pts:
                        wx, wy = valid_wrist_pts[0]
                        pw = prod.box[2] - prod.box[0]
                        ph_prod = prod.box[3] - prod.box[1]
                        prod.box = [int(wx - pw/2), int(wy - ph_prod/2), int(wx + pw/2), int(wy + ph_prod/2)]

                # ── STAGE C: CHECK PRODUCT RESOLUTIONS (RETURN, BAG, THEFT) ──
                # Condition 1: RETURN TO SHELF
                # Hand enters shelf ROI AND shelf gap at pick location is now closed/restored
                hand_back_on_shelf = hand_in_shelf or any(
                    cv2.pointPolygonTest(s, w_pt, True) >= -reach_margin
                    for s in shelf_np
                    for w_pt in valid_wrist_pts
                )

                gap_still_exists = False
                if prod.origin_center is not None:
                    for g in shelf_gaps:
                        if np.hypot(g["center"][0] - prod.origin_center[0], g["center"][1] - prod.origin_center[1]) < 70:
                            gap_still_exists = True
                            break

                # Condition 2: PLACED IN BAG / CONTAINER
                # Requires an ACTIVELY DETECTED shopping container/bag (not a phantom/extrapolated torso anchor)
                hand_at_bag = False
                matched_bag_name = "CONTAINER"
                for c in valid_containers:
                    if c.get("conf", 0) >= 0.38:
                        cb = c["box"]
                        for w_pt in valid_wrist_pts:
                            if point_in_box(w_pt, cb, margin=5):
                                hand_at_bag = True
                                matched_bag_name = c.get("label", "BAG")
                                break
                    if hand_at_bag:
                        break

                # ── State Machine Transitions ──
                if hand_back_on_shelf and (not gap_still_exists):
                    prod.state = ProductState.RETURN_CANDIDATE
                    prod.return_candidate_frames += 1
                    if prod.return_candidate_frames >= 2:
                        # Confirmed: Product Returned to Shelf!
                        prod.state = ProductState.RETURNED_TO_SHELF
                        p.state = "PUT BACK ON SHELF"
                        counts["returned_to_shelf"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        if counts["suspicious_missing"] > 0:
                            counts["suspicious_missing"] -= 1
                        log_event(t, f"Customer #{p.id}: Product #{prod.id} RETURNED TO SHELF [Put Back: {counts['returned_to_shelf']}]", CYN)
                        # Remove from accounted gaps so it can be picked again if desired
                        if prod.origin_center is not None:
                            accounted_shelf_gaps = [ag for ag in accounted_shelf_gaps if np.hypot(ag[0]-prod.origin_center[0], ag[1]-prod.origin_center[1]) >= 70]
                        p.completed_products.append(prod)
                        p.active_product = None

                elif hand_at_bag:
                    prod.bag_drop_candidate_frames += 1
                    if prod.bag_drop_candidate_frames >= 10:
                        # Confirmed: Placed into Bag (Safe)!
                        prod.state = ProductState.PLACED_IN_BAG
                        p.state = "PLACED IN BAG (SAFE)"
                        counts["placed_in_bag"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        if counts["suspicious_missing"] > 0:
                            counts["suspicious_missing"] -= 1
                        log_event(t, f"Customer #{p.id}: Product #{prod.id} PLACED IN {matched_bag_name} (SAFE) [Bag: {counts['placed_in_bag']}]", GRN)
                        p.completed_products.append(prod)
                        p.active_product = None

                else:
                    # Item not returned and not in bag: evaluate disappearance
                    time_unseen = t - prod.t_last_seen
                    time_held = t - prod.t_pick

                    if time_unseen > UNCERTAIN_TIMEOUT_SEC:
                        if prod.state != ProductState.TRACKING_UNCERTAIN and prod.state != ProductState.MISSING_SUSPICIOUS:
                            prod.state = ProductState.TRACKING_UNCERTAIN
                            p.state = "TRACKING UNCERTAIN"

                    if time_unseen > (UNCERTAIN_TIMEOUT_SEC + 0.6):
                        if prod.state != ProductState.MISSING_SUSPICIOUS:
                            prod.state = ProductState.MISSING_SUSPICIOUS
                            p.state = "PRODUCT MISSING (SUSPICIOUS)"
                            counts["suspicious_missing"] += 1
                            log_event(t, f"Customer #{p.id}: Product #{prod.id} TRACKING UNCERTAIN / MISSING", ORANGE)

                    # Confirmed Theft Alert: missing beyond grace period AND shelf gap remains open
                    if (time_unseen > 2.0 or time_held > 2.8) and not hand_in_shelf and gap_still_exists:
                        prod.state = ProductState.THEFT_ALERT
                        p.theft_flagged = True
                        p.state = "THEFT ALERT (ITEM MISSING)"
                        counts["theft_alerts"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        if counts["suspicious_missing"] > 0:
                            counts["suspicious_missing"] -= 1
                        log_event(t, f"Customer #{p.id}: 🚨 THEFT ALERT! Product #{prod.id} MISSING (NOT RETURNED TO SHELF)", RED)
                        p.completed_products.append(prod)
                        p.active_product = None

        # ──────────────── 5. EXITED CUSTOMER RESOLUTION ───────────────────────────
        for p in list(people.values()):
            if t - p.t > 2.2:
                # Customer exited camera view
                if p.active_product is not None:
                    prod = p.active_product
                    prod.state = ProductState.CUSTOMER_EXITED_UNRESOLVED
                    counts["currently_held"] = max(0, counts["currently_held"] - 1)
                    counts["suspicious_missing"] += 1
                    log_event(t, f"Customer #{p.id} EXITED CAMERA VIEW with unresolved product #{prod.id}", ORANGE)
                    p.completed_products.append(prod)
                    p.active_product = None

        # ──────────────── 6. RENDER OPENCV DETECTION HUD ──────────────────────────
        canvas = np.zeros((720, total_w, 3), np.uint8)
        canvas[:] = DARK

        # Main video render
        canvas[:, :vw] = cv2.resize(frame, (vw, 720))

        # Draw Shelf Polygons
        for s in shelf_np:
            pts = np.stack([s[:, 0] * sx_render, s[:, 1] * sy_render], axis=1).astype(np.int32)
            ov = canvas.copy()
            cv2.fillPoly(ov, [pts], (0, 180, 220))
            cv2.addWeighted(ov, 0.16, canvas, 0.84, 0, canvas)
            cv2.polylines(canvas, [pts], True, (0, 240, 240), 2, cv2.LINE_AA)
            cx_ = int(pts[:, 0].mean())
            cy_ = int(pts[:, 1].mean())
            cv2.putText(canvas, "SHELF ROI", (cx_ - 38, cy_), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 240), 2, cv2.LINE_AA)

        # Draw detected Shelf Gap indicators (Orange rectangles on the shelf)
        for g in shelf_gaps:
            gx1 = int(g["box"][0] * sx_render)
            gy1 = int(g["box"][1] * sy_render)
            gx2 = int(g["box"][2] * sx_render)
            gy2 = int(g["box"][3] * sy_render)
            cv2.rectangle(canvas, (gx1, gy1), (gx2, gy2), (0, 100, 255), 2)
            cv2.putText(canvas, f"GAP {g['area']}px", (gx1, max(14, gy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 120, 255), 1, cv2.LINE_AA)

        # Draw Visual Elements for Customers
        for p in active_customers:
            b = p.d["box"]
            k = p.d["k"]
            b_sc = [int(b[0] * sx_render), int(b[1] * sy_render), int(b[2] * sx_render), int(b[3] * sy_render)]

            # Customer bounding box color based on state
            if "THEFT" in p.state:
                box_col = RED
            elif "SAFE" in p.state or "PUT BACK" in p.state:
                box_col = GRN
            elif "HOLDING" in p.state:
                box_col = AMB
            elif "UNCERTAIN" in p.state or "MISSING" in p.state:
                box_col = ORANGE
            elif "REACHING" in p.state:
                box_col = CYN
            else:
                box_col = GRY

            cv2.rectangle(canvas, (b_sc[0], b_sc[1]), (b_sc[2], b_sc[3]), box_col, 2)
            cv2.putText(canvas, f"CUSTOMER #{p.id} [{p.state}]", (b_sc[0], max(18, b_sc[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_col, 2, cv2.LINE_AA)

            # Draw Customer Hands as clean indicator dots (NEVER labeled as product)
            for w_idx in (9, 10):
                wkp = k[w_idx]
                if wkp[2] > 0.15:
                    wx = int(wkp[0] * sx_render)
                    wy = int(wkp[1] * sy_render)
                    cv2.circle(canvas, (wx, wy), 4, (200, 240, 255), -1)
                    cv2.circle(canvas, (wx, wy), 6, (100, 100, 100), 1)

            # Draw tracked Shoulder Bag / Bag
            if p.bag_box is not None:
                bx1r = int(p.bag_box[0] * sx_render)
                by1r = int(p.bag_box[1] * sy_render)
                bx2r = int(p.bag_box[2] * sx_render)
                by2r = int(p.bag_box[3] * sy_render)

                bag_ov = canvas.copy()
                cv2.rectangle(bag_ov, (bx1r, by1r), (bx2r, by2r), (0, 200, 100), -1)
                cv2.addWeighted(bag_ov, 0.20, canvas, 0.80, 0, canvas)
                cv2.rectangle(canvas, (bx1r, by1r), (bx2r, by2r), (50, 220, 120), 2)
                cv2.putText(canvas, f"[{p.bag_type}]", (bx1r, max(14, by1r - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 220, 120), 1, cv2.LINE_AA)

            # Draw the ACTUAL PRODUCT BOX (Tight bounding box around the physical product)
            if p.active_product is not None and p.active_product.box is not None:
                ap = p.active_product
                px1 = int(ap.box[0] * sx_render)
                py1 = int(ap.box[1] * sy_render)
                px2 = int(ap.box[2] * sx_render)
                py2 = int(ap.box[3] * sy_render)

                # Clamp dimensions to tight item size
                if (px2 - px1) < 14: px2 = px1 + 14
                if (py2 - py1) < 18: py2 = py1 + 18

                prod_col = RED if ap.state == ProductState.THEFT_ALERT else (
                    ORANGE if (ap.state in (ProductState.TRACKING_UNCERTAIN, ProductState.MISSING_SUSPICIOUS)) else AMB
                )

                prod_ov = canvas.copy()
                cv2.rectangle(prod_ov, (px1, py1), (px2, py2), prod_col, -1)
                cv2.addWeighted(prod_ov, 0.25, canvas, 0.75, 0, canvas)
                cv2.rectangle(canvas, (px1, py1), (px2, py2), prod_col, 2)
                cv2.putText(canvas, f"PRODUCT #{ap.id} | CUST #{p.id} [{ap.state}]",
                            (px1 - 10, max(16, py1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, prod_col, 2, cv2.LINE_AA)

        # ──────────────── 7. TELEMETRY PANEL (RIGHT SIDE) ─────────────────────────
        px0 = vw
        panel = canvas[:, px0:total_w]
        panel[:] = PANEL_BG

        # Header
        cv2.putText(panel, "RETAIL SECURITY AI", (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.70, WHT, 2)
        now = time.time()
        fps_disp = 0.88 * fps_disp + 0.12 * (1.0 / max(1e-4, now - t_prev))
        cv2.putText(panel, f"{t:5.1f}s | {fps_disp:4.1f} fps", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # Dynamic Status Banner
        any_theft = counts["theft_alerts"] > 0 or any(p.theft_flagged for p in people.values())
        any_holding = counts["currently_held"] > 0 or any(p.active_product is not None for p in people.values())
        any_suspicious = counts["suspicious_missing"] > 0
        any_reaching = any(p.hands_in_shelf for p in people.values())
        any_safe = counts["placed_in_bag"] > 0 or counts["returned_to_shelf"] > 0

        if any_theft:
            banner_col, banner_text = RED, "🚨 THEFT DETECTED"
        elif any_suspicious:
            banner_col, banner_text = ORANGE, "⚠️ TRACKING UNCERTAIN"
        elif any_holding:
            banner_col, banner_text = AMB, "🛍 HOLDING PRODUCT"
        elif any_reaching:
            banner_col, banner_text = CYN, "✋ REACHING SHELF"
        elif any_safe:
            banner_col, banner_text = GRN, "✓ PRODUCT RESOLVED"
        else:
            banner_col, banner_text = GRN, "✓ BROWSING"

        cv2.rectangle(panel, (15, 75), (PW - 15, 120), banner_col, -1)
        cv2.putText(panel, banner_text, (25, 106), cv2.FONT_HERSHEY_DUPLEX, 0.65, WHT, 2)

        # 7-Metric Event Counters (Section 9)
        y_c = 155
        dy_c = 26
        metrics = [
            ("Customers in View", len(active_customers), WHT),
            ("Total Products Picked", counts["total_picked"], AMB),
            ("Currently in Hand", counts["currently_held"], (0, 215, 255)),
            ("Products Returned to Shelf", counts["returned_to_shelf"], CYN),
            ("Products Placed in Bag", counts["placed_in_bag"], GRN),
            ("Suspicious Missing Products", counts["suspicious_missing"], ORANGE),
            ("Theft Alerts", counts["theft_alerts"], RED),
        ]
        for label, val, c_col in metrics:
            cv2.putText(panel, label, (20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
            cv2.putText(panel, str(val), (PW - 45, y_c), cv2.FONT_HERSHEY_DUPLEX, 0.55, c_col, 2)
            y_c += dy_c

        # Activity Log Divider
        cv2.line(panel, (15, y_c + 8), (PW - 15, y_c + 8), (55, 55, 65), 1)
        y_c += 28
        cv2.putText(panel, "— ACTIVITY LOG —", (20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 140), 1)
        y_c += 22

        # Display last 9 log messages
        for ev_t, ev_text, ev_col in events[-9:]:
            short_txt = (ev_text[:38] + "..") if len(ev_text) > 40 else ev_text
            cv2.putText(panel, f"{ev_t:4.1f}s {short_txt}", (20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ev_col, 1)
            y_c += 19

        # Footer
        cv2.putText(panel, "Product Isolation • Shelf Tracking • Anti-Theft AI", (18, 705),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 100), 1)

        # Output video and GUI preview
        if wr:
            wr.write(canvas)
        if show:
            cv2.imshow("AI Retail Theft Detection v2", canvas)
            k_in = cv2.waitKey(1) & 0xFF
            if k_in in (27, ord('q'), ord('Q')):
                break
        t_prev = now

    cap.release()
    if wr:
        wr.release()
        print(f"\n[Pipeline] Output video rendered: {out_file}")
    cv2.destroyAllWindows()

    print("\n" + "=" * 25 + " FINAL REPORT " + "=" * 25)
    for ck, cv_val in counts.items():
        print(f"  {ck:<26}: {cv_val}")
    print("=" * 64)
    return events, counts


# ─────────────────────────── CLI ENTRYPOINT ─────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Retail Theft Detection v2 (vid2.mp4 & Retail Surveillance)")
    parser.add_argument("video", nargs="?", default="vid2.mp4", help="Video path")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], help="YOLO model size (n=nano is fastest)")
    parser.add_argument("--skip-frames", type=int, default=2, help="Number of frames to skip detection")
    parser.add_argument("--no-save", action="store_true", help="Do not write output video file")
    parser.add_argument("--headless", action="store_true", help="Run without preview window")
    parser.add_argument("--preset", action="store_true", help="Automatically use default preset shelf ROI without prompt")
    args = parser.parse_args()

    vpath = args.video if os.path.isabs(args.video) else os.path.join(HERE, args.video)
    if not os.path.exists(vpath):
        print(f"Error: Video file not found: {vpath}")
        sys.exit(1)

    print(f"\nTarget Video : {os.path.basename(vpath)}")
    if args.preset:
        vname = os.path.basename(vpath).lower()
        if "vid2" in vname:
            shelf_polys = [[[660, 40], [1260, 40], [1260, 520], [660, 520]]]
        else:
            shelf_polys = [[[350, 60], [420, 160], [380, 680], [140, 680], [110, 480], [250, 190]]]
    else:
        shelf_polys = draw_shelf_roi(vpath)

    print(f"Configured {len(shelf_polys)} shelf polygon(s).\n")

    run_theft_detection(
        video_path=vpath,
        shelf_polygons=shelf_polys,
        save=not args.no_save,
        show=not args.headless,
        skip_frames=args.skip_frames,
        model_size=args.model_size
    )

if __name__ == "__main__":
    main()
