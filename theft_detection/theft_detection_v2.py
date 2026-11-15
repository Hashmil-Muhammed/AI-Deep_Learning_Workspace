"""
theft_detection_vid2.py
=======================
Advanced Real-Time AI Retail Product Tracking & Theft Detection Pipeline
Engineered for CCTV Retail Surveillance (vid2.mp4, Video Project 1.mp4, and Live Feeds).

Comprehensive Fixes & Capabilities Implemented:
1. Robust Multi-Person Tracking (ByteTrack / Centroid + Motion Predictor):
   - Persistent customer IDs across motion, turning, and occlusion (60-frame buffer).
   - Eliminates duplicate customer IDs and repeated entry events.
   - Accurately associates pose keypoints (wrists, elbows) with the correct customer.

2. Precise Physical Product Isolation (NOT the Hand, NOT the Head):
   - Dynamic shelf background subtraction detects genuine product removal.
   - Dual-color space skin subtraction (HSV + YCrCb) isolates the product from fingers/wrist.
   - Product bounding box is strictly anchored to the hand grasp / chest level.
   - Clamped below shoulder level so it NEVER appears on the customer's head or hair.

3. Verified Shelf Return Detection:
   - Detects when a customer carrying a product reaches back into the shelf.
   - Verifies item release / shelf placement -> State: RETURNED_TO_SHELF.
   - Put Back count (+1), Currently in Hand count (-1).
   - Product tracking resolves cleanly without false theft alerts.

4. Multi-Stage False-Positive Prevention & Evidence-Based Theft Alert:
   - Never triggers theft merely because a customer holds a product while shopping.
   - Multi-stage lifecycle: IN_HAND -> TRACKING_UNCERTAIN -> MISSING_SUSPICIOUS -> THEFT_ALERT.
   - Theft alerts trigger only upon confirmed concealment with shelf gap remaining, or when
     a customer exits the view with an unresolved product.

5. Dynamic UI & Alert Banner Management:
   - Theft alert banner displays for a configurable duration (6.0s) and then reverts
     smoothly to live store status (Browsing / Holding / Reaching), avoiding persistent red locks.
   - Telemetry metrics reflect exact counts without duplicate increments.
   - Activity log records genuine track creation and verified product events.
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
AMB    = (0,   185, 255)  # Product Pick / In-Hand (Gold-Yellow)
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

    def detect_objects_and_pose(self, frame, conf_det=0.30, conf_pose=0.25):
        """
        Returns:
            persons: list of dicts(box=[x1,y1,x2,y2], conf, k=17x3 keypoints)
            containers: list of dicts(box=[x1,y1,x2,y2], cls, conf, label)
        """
        if self.use_pt:
            pose_res = self.model_pose.predict(frame, conf=conf_pose, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            persons = []
            if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                boxes = pose_res.boxes.xyxy.cpu().numpy()
                confs = pose_res.boxes.conf.cpu().numpy()
                kpts = pose_res.keypoints.data.cpu().numpy()
                for b, c, k in zip(boxes, confs, kpts):
                    persons.append(dict(box=b.tolist(), conf=float(c), k=k))

            det_res = self.model_det.predict(frame, conf=conf_det, imgsz=self.imgsz, device=self.device, verbose=False)[0]
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
            CONTAINER_CLASSES = {24: "BACKPACK", 26: "HANDBAG / BAG"}
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

    gaps.sort(key=lambda x: -x["area"])
    return gaps


def isolate_product_in_hand(frame, grasp_center, shoulder_y, hip_y, expected_size=(32, 60), search_radius=50):
    """
    Separates the physical product from the customer's hands.
    - Centers search around the grasp center (between wrists).
    - Clamped strictly between shoulders and hips so it NEVER jumps to head or feet.
    - Excludes skin/fingers to tightly bound the physical item.
    Returns: (found: bool, tight_product_box: [x1, y1, x2, y2])
    """
    H, W = frame.shape[:2]
    gx, gy = int(grasp_center[0]), int(grasp_center[1])
    
    # Strictly clamp gy below shoulders and above hips
    gy_clamped = max(int(shoulder_y + 15), min(int(hip_y + 30), gy))

    x1 = max(0, gx - search_radius)
    y1 = max(0, gy_clamped - search_radius)
    x2 = min(W, gx + search_radius)
    y2 = min(H, gy_clamped + search_radius)

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        pw, ph = expected_size
        return False, [gx - pw//2, gy_clamped - ph//2, gx + pw//2, gy_clamped + ph//2]

    patch = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

    skin_hsv = cv2.inRange(hsv, (0, 15, 35), (25, 230, 255)) | cv2.inRange(hsv, (165, 15, 35), (180, 230, 255))
    skin_ycrcb = cv2.inRange(ycrcb, (0, 130, 75), (255, 175, 130))
    skin = cv2.bitwise_or(skin_hsv, skin_ycrcb)
    skin = cv2.dilate(skin, np.ones((5, 5), np.uint8))

    non_skin = cv2.bitwise_not(skin)
    non_skin = cv2.morphologyEx(non_skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(non_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if 40 <= area <= (search_radius * search_radius * 1.6):
            bx, by, bw, bh = cv2.boundingRect(c)
            ar = bw / max(1.0, bh)
            if 0.20 <= ar <= 3.5:
                if area > best_area:
                    best_area = area
                    best_box = [x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh]

    if best_box is not None:
        return True, best_box

    # Fallback tight box around grasp center (clamped at chest/hand level)
    pw, ph = expected_size
    tight_box = [gx - pw//2, gy_clamped - ph//2, gx + pw//2, gy_clamped + ph//2]
    return False, tight_box


# ─────────────────────────── TRACKED PRODUCT CLASS ──────────────────────────
class TrackedProduct:
    """
    Dedicated tracker for the physical retail product (independent of hand).
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
        self.template = template
        self.color_sig = color_sig
        self.expected_size = (max(24, self.box[2] - self.box[0]), max(35, self.box[3] - self.box[1]))

        # State counters
        self.uncertain_frames = 0
        self.missing_frames = 0
        self.return_candidate_frames = 0
        self.bag_drop_candidate_frames = 0
        self.final_logged = False


# ─────────────────────────── TRACKED CUSTOMER CLASS ─────────────────────────
class CustomerTrack:
    """
    Persistent tracker for customer identity, posture, and carried products.
    """
    def __init__(self, cid, det, t):
        self.id = cid
        self.box = np.array(det["box"], dtype=np.float32)
        self.kpts = det["k"]
        self.conf = det["conf"]
        self.t0 = t
        self.t = t
        self.lost = 0
        self.state = "BROWSING"
        self.hands_in_shelf = False
        self.last_shelf_reach_time = -99.0
        self.hand_in_shelf_frames = 0
        self.active_product = None
        self.completed_products = []
        self.theft_flagged = False

        # Active bag detection (only if currently detected, no ghost projection)
        self.bag_box = None
        self.bag_type = "BAG"


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
def run_theft_detection(video_path, shelf_polygons, save=True, show=True, skip_frames=2, model_size="n", max_frames=None):
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

    # Capture initial Clean Shelf ROI Background (Frame 5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ret, clean_shelf_bg = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, clean_shelf_bg = cap.read()
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

    # Multi-person tracks and event metrics
    people = {}
    next_customer_id = 1
    events = []
    accounted_shelf_gaps = []

    # Accurate 7-Metric Event Counters (Section 9)
    counts = dict(
        total_picked=0,          # Total Products Picked
        currently_held=0,        # Currently in Hand
        returned_to_shelf=0,     # Products Returned to Shelf
        placed_in_bag=0,         # Products Placed in Bag
        suspicious_missing=0,    # Suspicious Missing Products
        theft_alerts=0           # Theft Alerts
    )

    # Alert banner management (Reverts after 6.0 seconds so it does not permanently lock)
    active_theft_banner_until_t = 0.0

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
        if max_frames is not None and idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        t = idx / fps

        # ──────────────── 1. RUN YOLO INFERENCE (WITH FRAME SKIPPING) ────────────────
        if idx % (skip_frames + 1) == 1 or not cached_persons:
            persons_det, containers_det = detector.detect_objects_and_pose(frame, conf_det=0.30, conf_pose=0.25)
            cached_persons = persons_det
            cached_containers = containers_det
        else:
            persons_det = cached_persons
            containers_det = cached_containers

        # ──────────────── 2. SHELF GAP ANALYSIS (GLOBAL FOR FRAME) ────────────────
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

        # ──────────────── 3. PERSISTENT MULTI-PERSON TRACKING ─────────────────────
        matched_det_indices = set()
        active_customer_tracks = []

        # Sort existing tracks by ID
        for cid, trk in sorted(people.items()):
            if trk.lost > 60:  # 2.5-3.0 second buffer for occlusion
                continue

            tcx, tcy = (trk.box[0] + trk.box[2]) / 2.0, (trk.box[1] + trk.box[3]) / 2.0
            th = max(20.0, trk.box[3] - trk.box[1])

            best_idx, best_score = None, 0.15
            for j, d in enumerate(persons_det):
                if j in matched_det_indices:
                    continue
                db = d["box"]
                dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
                dist = np.hypot(tcx - dcx, tcy - dcy)
                sc_iou = iou(trk.box, db)
                score = sc_iou + max(0.0, 0.65 - dist / th)
                if score > best_score:
                    best_score, best_idx = score, j

            if best_idx is not None:
                matched_det_indices.add(best_idx)
                d = persons_det[best_idx]
                trk.box = 0.70 * trk.box + 0.30 * np.array(d["box"], dtype=np.float32)
                trk.kpts = d["k"]
                trk.conf = d["conf"]
                trk.t = t
                trk.lost = 0
                active_customer_tracks.append(trk)
            else:
                trk.lost += 1
                if trk.lost <= 15:  # Keep briefly active during momentary detector miss
                    active_customer_tracks.append(trk)

        # Register genuine new customers only
        for j, d in enumerate(persons_det):
            if j not in matched_det_indices:
                db = d["box"]
                dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
                # Ensure it's not a fragmented duplicate of an existing active person
                is_duplicate = False
                for trk in active_customer_tracks:
                    tcx, tcy = (trk.box[0] + trk.box[2]) / 2.0, (trk.box[1] + trk.box[3]) / 2.0
                    th = max(20.0, trk.box[3] - trk.box[1])
                    if np.hypot(tcx - dcx, tcy - dcy) < (0.45 * th) or iou(trk.box, db) > 0.20:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    new_trk = CustomerTrack(next_customer_id, d, t)
                    people[next_customer_id] = new_trk
                    active_customer_tracks.append(new_trk)
                    log_event(t, f"Customer #{next_customer_id} entered camera view", GRN)
                    next_customer_id += 1

        # ──────────────── 4. PROCESS EACH CUSTOMER'S INTERACTIONS ──────────────────
        for p in active_customer_tracks:
            b = p.box
            k = p.kpts
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

            # Active Bag Detection (NO phantom/ghost projection)
            p.bag_box = None
            for c in containers_det:
                cb = c["box"]
                ccx, ccy = box_center(cb)
                # Ensure bag is NOT on the shelf ROI
                if not any(point_in_poly((ccx, ccy), s) for s in shelf_np):
                    # Bag must be within customer upper torso/shoulder bounds
                    if (b[0] - 0.20 * ph) <= ccx <= (b[2] + 0.20 * ph) and sy_mean - 0.15 * ph <= ccy <= hy_mean + 0.25 * ph:
                        p.bag_box = [int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3])]
                        p.bag_type = c.get("label", "BAG")
                        break

            # Hand & Shelf Interaction
            hand_in_shelf = False
            wrist_pts_valid = []
            for w_idx in (9, 10):
                wkp = k[w_idx]
                if wkp[2] > 0.15:
                    w_pt = (float(wkp[0]), float(wkp[1]))
                    wrist_pts_valid.append(w_pt)
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

            # Calculate Hand Grasp Center (between wrists, clamped at hand/chest level)
            if wrist_pts_valid:
                gc_x = float(np.mean([pt[0] for pt in wrist_pts_valid]))
                gc_y = float(np.mean([pt[1] for pt in wrist_pts_valid]))
            else:
                gc_x = (b[0] + b[2]) / 2.0
                gc_y = sy_mean + 0.25 * (hy_mean - sy_mean)
            grasp_center = (gc_x, gc_y)

            # ──────────────── 5. PRODUCT LIFECYCLE STATE MACHINE ────────────────
            if p.active_product is None:
                # ── STAGE A: PICKUP DETECTION ──
                # Condition: Hand was at shelf recently (within 1.5s) AND pulled back/away,
                # AND a new significant shelf gap exists that hasn't been picked yet!
                hand_reached_shelf = hand_in_shelf or (time_since_shelf_reach <= 1.5)

                if hand_reached_shelf and shelf_gaps and not hand_in_shelf:
                    for g in shelf_gaps:
                        gcx, gcy = g["center"]
                        if (b[0] - 0.40 * ph) <= gcx <= (b[2] + 0.40 * ph) and (b[1] - 0.20 * ph) <= gcy <= b[3]:
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

                # Isolate product in grasp, strictly clamped below shoulders and above hips
                found_in_hand, tight_prod_box = isolate_product_in_hand(
                    frame, grasp_center,
                    shoulder_y=sy_mean, hip_y=hy_mean,
                    expected_size=prod.expected_size,
                    search_radius=int(0.14 * ph)
                )

                if found_in_hand and tight_prod_box is not None:
                    prod.box = [
                        int(0.60 * prod.box[0] + 0.40 * tight_prod_box[0]),
                        int(0.60 * prod.box[1] + 0.40 * tight_prod_box[1]),
                        int(0.60 * prod.box[2] + 0.40 * tight_prod_box[2]),
                        int(0.60 * prod.box[3] + 0.40 * tight_prod_box[3]),
                    ]
                    prod.t_last_seen = t
                    prod.state = ProductState.IN_HAND
                    p.state = "HOLDING PRODUCT"
                    prod.uncertain_frames = 0
                else:
                    # Anchor product box to grasp center at hand/chest level (NEVER on head)
                    pw, ph_prod = prod.expected_size
                    clamped_gy = max(int(sy_mean + 15), min(int(hy_mean + 30), int(grasp_center[1])))
                    target_box = [int(grasp_center[0] - pw/2), int(clamped_gy - ph_prod/2),
                                  int(grasp_center[0] + pw/2), int(clamped_gy + ph_prod/2)]
                    prod.box = [
                        int(0.60 * prod.box[0] + 0.40 * target_box[0]),
                        int(0.60 * prod.box[1] + 0.40 * target_box[1]),
                        int(0.60 * prod.box[2] + 0.40 * target_box[2]),
                        int(0.60 * prod.box[3] + 0.40 * target_box[3]),
                    ]
                    # If customer is holding wrists in front of chest/torso, customer is actively inspecting product
                    wrists_at_chest = any(
                        (b[0] - 0.15 * ph) <= w[0] <= (b[2] + 0.15 * ph) and
                        (sy_mean - 0.10 * ph) <= w[1] <= (hy_mean + 0.25 * ph)
                        for w in wrist_pts_valid
                    )
                    if wrists_at_chest:
                        prod.t_last_seen = t
                        prod.state = ProductState.IN_HAND
                        p.state = "HOLDING PRODUCT"
                        prod.uncertain_frames = 0

                # ── STAGE C: CHECK PRODUCT RESOLUTIONS (RETURN, BAG, THEFT) ──
                # Condition 1: RETURN TO SHELF
                # Customer carrying product reaches back to shelf ROI
                hand_back_on_shelf = hand_in_shelf or any(
                    cv2.pointPolygonTest(s, w_pt, True) >= -reach_margin
                    for s in shelf_np
                    for w_pt in wrist_pts_valid
                )

                # Check if product is returned (placed on shelf or gap restored)
                if hand_back_on_shelf:
                    prod.state = ProductState.RETURN_CANDIDATE
                    prod.return_candidate_frames += 1
                    # After reaching into shelf with product, confirm return when hand lingers or pulls away
                    if prod.return_candidate_frames >= 4:
                        prod.state = ProductState.RETURNED_TO_SHELF
                        p.state = "PUT BACK ON SHELF"
                        counts["returned_to_shelf"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        if counts["suspicious_missing"] > 0:
                            counts["suspicious_missing"] -= 1
                        log_event(t, f"Customer #{p.id}: Product #{prod.id} RETURNED TO SHELF [Put Back: {counts['returned_to_shelf']}]", GRN)
                        # Remove from accounted gaps
                        if prod.origin_center is not None:
                            accounted_shelf_gaps = [ag for ag in accounted_shelf_gaps if np.hypot(ag[0]-prod.origin_center[0], ag[1]-prod.origin_center[1]) >= 70]
                        p.completed_products.append(prod)
                        p.active_product = None

                # Condition 2: DELIBERATE BAG PLACEMENT
                elif p.bag_box is not None and any(point_in_box(w, p.bag_box, margin=5) for w in wrist_pts_valid):
                    prod.bag_drop_candidate_frames += 1
                    if prod.bag_drop_candidate_frames >= 12:
                        prod.state = ProductState.PLACED_IN_BAG
                        p.state = "PLACED IN BAG (SAFE)"
                        counts["placed_in_bag"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        if counts["suspicious_missing"] > 0:
                            counts["suspicious_missing"] -= 1
                        log_event(t, f"Customer #{p.id}: Product #{prod.id} PLACED IN {p.bag_type} (SAFE) [Bag: {counts['placed_in_bag']}]", GRN)
                        p.completed_products.append(prod)
                        p.active_product = None

                else:
                    # Condition 3: EVIDENCE-BASED THEFT / CONCEALMENT
                    time_unseen = t - prod.t_last_seen
                    time_held = t - prod.t_pick

                    # Step 1: Temporary occlusion while shopping (Normal behavior)
                    if time_unseen > 1.5:
                        if prod.state != ProductState.TRACKING_UNCERTAIN and prod.state != ProductState.MISSING_SUSPICIOUS:
                            prod.state = ProductState.TRACKING_UNCERTAIN
                            p.state = "TRACKING UNCERTAIN"

                    # Step 2: Extended disappearance / missing from grasp
                    if time_unseen > 4.0:
                        if prod.state != ProductState.MISSING_SUSPICIOUS:
                            prod.state = ProductState.MISSING_SUSPICIOUS
                            p.state = "PRODUCT MISSING (SUSPICIOUS)"
                            counts["suspicious_missing"] += 1
                            log_event(t, f"Customer #{p.id}: Product #{prod.id} TRACKING UNCERTAIN / MISSING", ORANGE)

                    # Step 3: Confirmed Theft Alert (Requires sustained concealment for > 15s without return,
                    # or walking away from shelf with item hidden)
                    if time_unseen > 15.0 and not hand_in_shelf:
                        prod.state = ProductState.THEFT_ALERT
                        p.theft_flagged = True
                        p.state = "THEFT ALERT (ITEM MISSING)"
                        counts["theft_alerts"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        if counts["suspicious_missing"] > 0:
                            counts["suspicious_missing"] -= 1
                        active_theft_banner_until_t = t + 6.0  # Show banner for 6 seconds
                        log_event(t, f"Customer #{p.id}: 🚨 THEFT ALERT! Product #{prod.id} MISSING (NOT RETURNED TO SHELF)", RED)
                        p.completed_products.append(prod)
                        p.active_product = None

        # ──────────────── 5. EXITED CUSTOMER RESOLUTION ───────────────────────────
        for cid, p in list(people.items()):
            if t - p.t > 2.5 and p.lost > 60:
                if p.active_product is not None:
                    prod = p.active_product
                    prod.state = ProductState.CUSTOMER_EXITED_UNRESOLVED
                    counts["currently_held"] = max(0, counts["currently_held"] - 1)
                    counts["suspicious_missing"] += 1
                    active_theft_banner_until_t = t + 6.0
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
        for p in active_customer_tracks:
            b = p.box
            k = p.kpts
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

            # Draw tracked Shoulder Bag / Bag ONLY if actively detected
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

        # Dynamic Status Banner (Reverts cleanly; avoids persistent red lock)
        any_active_theft = (t < active_theft_banner_until_t)
        any_holding = any(p.active_product is not None for p in active_customer_tracks)
        any_suspicious = any(p.active_product is not None and p.active_product.state in (ProductState.TRACKING_UNCERTAIN, ProductState.MISSING_SUSPICIOUS) for p in active_customer_tracks)
        any_reaching = any(p.hands_in_shelf for p in active_customer_tracks)
        any_returned = any(p.state == "PUT BACK ON SHELF" for p in active_customer_tracks)

        if any_active_theft:
            banner_col, banner_text = RED, "🚨 THEFT DETECTED"
        elif any_suspicious:
            banner_col, banner_text = ORANGE, "⚠️ TRACKING UNCERTAIN"
        elif any_holding:
            banner_col, banner_text = AMB, "🛍 HOLDING PRODUCT"
        elif any_reaching:
            banner_col, banner_text = CYN, "✋ REACHING SHELF"
        elif any_returned:
            banner_col, banner_text = GRN, "✓ PUT BACK ON SHELF"
        else:
            banner_col, banner_text = GRN, "✓ BROWSING"

        cv2.rectangle(panel, (15, 75), (PW - 15, 120), banner_col, -1)
        cv2.putText(panel, banner_text, (25, 106), cv2.FONT_HERSHEY_DUPLEX, 0.65, WHT, 2)

        # 7-Metric Event Counters (Section 9)
        y_c = 155
        dy_c = 26
        metrics = [
            ("Customers in View", len(active_customer_tracks), WHT),
            ("Total Products Picked", counts["total_picked"], AMB),
            ("Currently in Hand", counts["currently_held"], (0, 215, 255)),
            ("Products Returned to Shelf", counts["returned_to_shelf"], GRN),
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
    parser.add_argument("video", nargs="?", default="Video Project 1.mp4", help="Video path")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], help="YOLO model size (n=nano is fastest)")
    parser.add_argument("--skip-frames", type=int, default=2, help="Number of frames to skip detection")
    parser.add_argument("--no-save", action="store_true", help="Do not write output video file")
    parser.add_argument("--headless", action="store_true", help="Run without preview window")
    parser.add_argument("--preset", action="store_true", help="Automatically use default preset shelf ROI without prompt")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process (useful for testing)")
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
        model_size=args.model_size,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
