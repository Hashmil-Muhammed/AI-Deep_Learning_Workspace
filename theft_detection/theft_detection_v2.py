"""
theft_detection_vid2.py
=======================
Advanced Retail Theft Detection Pipeline (Optimized for vid2.mp4 & Retail CCTVs):

Key Capabilities:
1. Accurate Shoulder Bag & Handheld Bag Detection & Tracking:
   - Reliably detects shoulder bags ("thollil ulla bag"), backpacks, and handheld covers.
   - Automatically anchors shoulder bags to customer shoulder/torso keypoints.
   - Smoothly tracks the shoulder bag across frames without jumping or drifting down to legs.

2. Elimination of False Product Pick (Empty Hand Protection):
   - Entering or browsing the shelf with a bare hand is recognized as BROWSING / REACHING SHELF.
   - Does NOT declare an empty hand as a product.
   - Accurately checks palm region with HSV + YCrCb skin masking and shelf background difference.
   - Requires verified non-skin foreign object before confirming TOOK PRODUCT.

3. Complete Product Destination Resolution & State Machine:
   - TOOK PRODUCT: Hand in shelf takes an item -> Currently in Hand (+1), Took Product (+1).
   - RETURNED TO SHELF: Hand returns product to shelf -> Put Back on Shelf (+1), Currently in Hand (-1).
   - PLACED IN COVER / BAG: Hand puts product into shoulder bag / cover -> Placed in Cover/Bag (+1), Currently in Hand (-1).
   - THEFT ALERT: Product taken from shelf, not in hand, not in bag, not returned to shelf (missing/concealed)
     -> THEFT ALERT (+1), Currently in Hand (-1).

4. Real-time Telemetry HUD:
   - Live counts: Took Product, Currently in Hand, Placed in Bag, Put Back on Shelf, Theft Alerts.
   - Visual shelf ROI, tracked shoulder bag, hand status, and event activity log.
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
RED   = (40,  40,  230)
GRN   = (50,  205, 50)
AMB   = (0,   180, 255)
WHT   = (255, 255, 255)
CYN   = (220, 220, 0)
PRP   = (200, 60,  200)
BLK   = (0,   0,   0)
DARK  = (22,  22,  26)

SKEL_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

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
            containers: list of dicts(box=[x1,y1,x2,y2], cls, conf, label) (handbags, backpacks, covers)
        """
        if self.use_pt:
            # 1. Pose estimation
            pose_res = self.model_pose.predict(frame, conf=conf_pose, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            persons = []
            if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                boxes = pose_res.boxes.xyxy.cpu().numpy()
                confs = pose_res.boxes.conf.cpu().numpy()
                kpts = pose_res.keypoints.data.cpu().numpy()
                for b, c, k in zip(boxes, confs, kpts):
                    persons.append(dict(box=b.tolist(), conf=float(c), k=k))

            # 2. General object detection with low conf (0.15) for high recall on bags
            det_res = self.model_det.predict(frame, conf=conf_det, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            containers = []
            if det_res.boxes is not None and len(det_res.boxes) > 0:
                d_boxes = det_res.boxes.xyxy.cpu().numpy()
                d_confs = det_res.boxes.conf.cpu().numpy()
                d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
                # COCO classes: 24: backpack, 26: handbag, 28: suitcase
                CONTAINER_CLASSES = {24: "BACKPACK", 26: "HANDBAG / BAG", 28: "SUITCASE / BASKET"}
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
            CONTAINER_CLASSES = {24: "BACKPACK", 26: "HANDBAG / BAG", 28: "SUITCASE / BASKET"}
            for i in keep_d:
                c_id = int(cls_d[i])
                if c_id in CONTAINER_CLASSES:
                    x, y, bw, bh = bx_d[i]
                    containers.append(dict(box=[x, y, x + bw, y + bh], conf=float(sc_d[i]), cls=c_id, label=CONTAINER_CLASSES[c_id]))
        return persons, containers


# ─────────────────────────── UTILITY FUNCTIONS ──────────────────────────────
def iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def point_in_box(pt, box, margin=0):
    return (box[0] - margin <= pt[0] <= box[2] + margin and
            box[1] - margin <= pt[1] <= box[3] + margin)


def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


# ──────────────────────── HAND & PRODUCT ANALYSIS ───────────────────────────
def analyze_hand_patch(img, palm_pt, padding_size=24, bg_patch=None):
    """
    Carefully inspects the palm area for genuine foreign products.
    Differentiates bare skin from products (cans, bottles, boxes, packaging).
    """
    H, W = img.shape[:2]
    x, y = int(palm_pt[0]), int(palm_pt[1])
    x1 = max(0, x - padding_size)
    y1 = max(0, y - padding_size)
    x2 = min(W, x + padding_size)
    y2 = min(H, y + padding_size)

    if (x2 - x1) < 12 or (y2 - y1) < 12:
        return False, None, None, 0.0, None, 0.0

    patch = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

    # 1. Skin tone detector (HSV + YCrCb intersection)
    hsv_skin = cv2.inRange(hsv, (0, 25, 45), (25, 190, 255)) | cv2.inRange(hsv, (170, 25, 45), (180, 190, 255))
    ycrcb_skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_mask = cv2.bitwise_and(hsv_skin, ycrcb_skin)

    total_pixels = (x2 - x1) * (y2 - y1)
    skin_pixels = np.sum(skin_mask > 0)
    skin_ratio = skin_pixels / total_pixels

    # 2. Difference from clean shelf background if hand is in shelf
    diff_mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    if bg_patch is not None and bg_patch.shape == patch.shape:
        diff = cv2.absdiff(patch, bg_patch)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_mask = (diff_gray > 28).astype(np.uint8) * 255

    # 3. Vivid / product packaging colors
    vivid_mask = cv2.inRange(hsv, (0, 48, 40), (180, 255, 255))

    # Candidate object pixels: changed from shelf background or vivid non-skin
    if bg_patch is not None:
        candidate = cv2.bitwise_or(vivid_mask, diff_mask)
    else:
        candidate = vivid_mask

    # Must be non-skin
    object_mask = cv2.bitwise_and(candidate, cv2.bitwise_not(skin_mask))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, None, None, 0.0, None, skin_ratio

    # Find the largest non-skin object blob
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    ratio = area / total_pixels

    # Genuine product condition:
    # Meaningful non-skin blob area, non-skin ratio >= 4%, and skin does not completely dominate hand
    if area >= 20 and ratio >= 0.045 and skin_ratio < 0.85:
        bx, by, bw, bh = cv2.boundingRect(c)
        global_box = [x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh]
        mean_bgr = cv2.mean(patch, mask=object_mask)[:3]
        return True, object_mask, global_box, ratio, mean_bgr, skin_ratio

    return False, None, None, ratio, None, skin_ratio


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

    # Pre-configured default shelves for known videos
    vname = os.path.basename(video_path).lower()
    default_shelves = []
    if "vid2" in vname:
        # Default right shelf for vid2.mp4
        default_shelves = [[[660, 40], [1260, 40], [1260, 520], [660, 520]]]
    elif "project 1" in vname or "video project 1" in vname:
        # Default left shelf for Video Project 1.mp4
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
    print(" • Left-Click  : Place corner of shelf")
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
        for p in cur:
            cv2.circle(vis, p, 5, (0, 200, 255), -1)
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


# ─────────────────────────── TRACKED PERSON CLASS ───────────────────────────
class CustomerTrack:
    def __init__(self, pid, d, t):
        self.id = pid
        self.d = d
        self.t0 = t
        self.t = t
        self.state = "BROWSING"   # BROWSING, REACHING SHELF, HOLDING PRODUCT, PLACED IN COVER (SAFE), PUT BACK ON SHELF, THEFT ALERT
        self.hands_in_shelf = False
        self.holding_item = False
        self.held_items_count = 0
        self.item_color_bgr = None
        self.item_box = None
        self.item_took_time = None
        self.last_item_seen_time = -99.0
        self.theft_flagged = False

        # Bag / Cover Tracking
        self.bag_box = None              # Smooth bounding box [x1, y1, x2, y2]
        self.bag_type = "SHOULDER BAG"   # "SHOULDER BAG" or "HANDHELD BAG"
        self.bag_rel_offset = None       # (rel_cx, rel_cy, w, h) relative to torso
        self.bag_missing_frames = 0      # frames since last YOLO detection
        self.torso_anchor_prev = None

        # Debouncing counters
        self.pick_candidate_frames = 0
        self.return_candidate_frames = 0
        self.bag_drop_candidate_frames = 0
        self.disappear_candidate_frames = 0
        self.hand_in_shelf_counter = 0


# ─────────────────────────── MAIN DETECTION LOOP ────────────────────────────
def run_theft_detection(video_path, shelf_polygons, save=True, show=True, skip_frames=2, model_size="n"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Responsive window sizing
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
    for p in shelf_np:
        cv2.fillPoly(shelf_mask, [p], 255)

    # Capture initial Clean Shelf ROI Background
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read frame 0 for background reference.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    clean_shelf_bg = first_frame.copy()
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

    people = {}
    nid = 1
    events = []
    counts = dict(took_product=0, placed_cart=0, put_back=0, theft_suspicion=0, currently_held=0)

    def log_event(t, text, col):
        events.append((t, text, col))
        prefix = "🚨 " if col is RED else ("🛍 " if col is AMB else "✓ ")
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

        # 1. Run YOLO detection every (skip_frames + 1) frames
        if idx % (skip_frames + 1) == 1 or not cached_persons:
            persons_det, containers_det = detector.detect_objects_and_pose(frame, conf_det=0.15, conf_pose=0.25)
            cached_persons = persons_det
            cached_containers = containers_det
        else:
            persons_det = cached_persons
            containers_det = cached_containers

        # 2. Track Persons across frames
        used = set()
        for p in sorted(people.values(), key=lambda x: -x.t):
            if t - p.t > 1.8:
                continue
            best_idx, best_sc = None, 0.12
            for j, d in enumerate(persons_det):
                if j in used: continue
                sc = iou(p.d["box"], d["box"])
                if sc > best_sc:
                    best_idx, best_sc = j, sc
            if best_idx is not None:
                used.add(best_idx)
                p.d = persons_det[best_idx]
                p.t = t

        for j, d in enumerate(persons_det):
            if j not in used:
                people[nid] = CustomerTrack(nid, d, t)
                nid += 1

        active = [p for p in people.values() if p.t == t]

        # 3. Process Each Active Customer
        for p in active:
            b = p.d["box"]
            k = p.d["k"]
            ph = b[3] - b[1]
            pad_size = max(22, int(0.11 * ph))

            ls, rs = k[5], k[6]   # Shoulders
            le, re = k[7], k[8]   # Elbows
            lw, rw = k[9], k[10]  # Wrists
            lh, rh = k[11], k[12] # Hips

            # Calculate torso center and dimensions
            sh_valid = [q for q in (ls, rs) if q[2] > 0.15]
            hip_valid = [q for q in (lh, rh) if q[2] > 0.15]

            sy_mean = float(np.mean([q[1] for q in sh_valid])) if sh_valid else b[1] + 0.18 * ph
            hy_mean = float(np.mean([q[1] for q in hip_valid])) if hip_valid else sy_mean + 0.42 * ph
            torso_h = max(35.0, hy_mean - sy_mean)
            torso_cx = float(np.mean([q[0] for q in sh_valid + hip_valid])) if (sh_valid + hip_valid) else (b[0] + b[2]) / 2.0
            torso_cy = (sy_mean + hy_mean) / 2.0
            torso_w = max(40.0, (b[2] - b[0]) * 0.70)

            # ──────────────────── SHOULDER BAG / BAG TRACKING ────────────────────
            # Match containers associated with this customer
            c_cand = []
            for c in containers_det:
                cb = c["box"]
                cx = (cb[0] + cb[2]) / 2.0
                cy = (cb[1] + cb[3]) / 2.0
                if b[0] - 0.25 * ph <= cx <= b[2] + 0.25 * ph and b[1] - 0.10 * ph <= cy <= b[3]:
                    # Container is on/with customer
                    area_c = (cb[2] - cb[0]) * (cb[3] - cb[1])
                    c_cand.append((cb, c["conf"], c.get("label", "BAG"), area_c, cy))

            if c_cand:
                # Pick the most prominent bag on customer
                best_cb, best_conf, best_lbl, best_area, best_cy = max(c_cand, key=lambda x: x[1])
                is_torso_shoulder = (best_cy <= hy_mean + 0.25 * torso_h)
                detected_bag_box = [int(best_cb[0]), int(best_cb[1]), int(best_cb[2]), int(best_cb[3])]

                if p.bag_box is None:
                    p.bag_box = detected_bag_box
                else:
                    # Smooth EMA update
                    p.bag_box = [
                        int(0.65 * p.bag_box[0] + 0.35 * detected_bag_box[0]),
                        int(0.65 * p.bag_box[1] + 0.35 * detected_bag_box[1]),
                        int(0.65 * p.bag_box[2] + 0.35 * detected_bag_box[2]),
                        int(0.65 * p.bag_box[3] + 0.35 * detected_bag_box[3]),
                    ]
                p.bag_type = "SHOULDER BAG" if is_torso_shoulder else "HANDHELD BAG"
                p.bag_missing_frames = 0

                # Compute relative anchor to torso
                bag_cx = (p.bag_box[0] + p.bag_box[2]) / 2.0
                bag_cy = (p.bag_box[1] + p.bag_box[3]) / 2.0
                rel_x = (bag_cx - torso_cx) / torso_w
                rel_y = (bag_cy - torso_cy) / torso_h
                bag_w = p.bag_box[2] - p.bag_box[0]
                bag_h = p.bag_box[3] - p.bag_box[1]
                p.bag_rel_offset = (rel_x, rel_y, bag_w, bag_h)

            elif p.bag_box is not None and p.bag_rel_offset is not None and p.bag_missing_frames < 90:
                # Retain shoulder bag securely using torso/shoulder anchor
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

            # ──────────────── HAND & SHELF INTERACTION ANALYSIS ────────────────
            hand_in_shelf = False
            object_in_hand = False
            detected_obj_box = None
            obj_color = None
            reach_margin = int(0.06 * ph)

            for w_idx, e_idx in ((9, 7), (10, 8)):  # (Left Wrist, Left Elbow), (Right Wrist, Right Elbow)
                w = k[w_idx]
                if w[2] < 0.15:
                    continue
                w_pt = (int(w[0]), int(w[1]))

                # Calculate palm position extending away from sleeve along forearm vector
                e = k[e_idx]
                if e[2] > 0.15:
                    vec = w[:2] - e[:2]
                    mag = np.linalg.norm(vec)
                    uvec = vec / mag if mag > 1e-3 else np.array([0.0, 1.0])
                    palm_center = w[:2] + uvec * min(24.0, max(12.0, mag * 0.35))
                else:
                    palm_center = w[:2]
                palm_pt = (int(palm_center[0]), int(palm_center[1]))

                # Check if hand/palm is inside or reaching shelf ROI
                in_shelf = any(cv2.pointPolygonTest(s, (float(palm_pt[0]), float(palm_pt[1])), True) >= -reach_margin for s in shelf_np)
                if in_shelf:
                    hand_in_shelf = True

                # Extract shelf background reference patch
                x1_bg = max(0, palm_pt[0] - pad_size)
                y1_bg = max(0, palm_pt[1] - pad_size)
                x2_bg = min(W, palm_pt[0] + pad_size)
                y2_bg = min(H, palm_pt[1] + pad_size)
                bg_crop = clean_shelf_bg[y1_bg:y2_bg, x1_bg:x2_bg] if (x2_bg > x1_bg and y2_bg > y1_bg) else None

                # Analyze palm patch: verify if non-skin foreign object exists
                has_obj, o_mask, o_box, o_ratio, o_col, s_ratio = analyze_hand_patch(
                    frame, palm_pt, padding_size=pad_size, bg_patch=bg_crop
                )

                if has_obj and s_ratio < 0.85:
                    object_in_hand = True
                    detected_obj_box = o_box
                    obj_color = o_col

            p.hands_in_shelf = hand_in_shelf

            if hand_in_shelf:
                p.hand_in_shelf_counter += 1
            else:
                p.hand_in_shelf_counter = max(0, p.hand_in_shelf_counter - 1)

            # ──────────────────── STATE MACHINE & DESTINATION LOGIC ────────────────────
            # 1. Product Pick Detection: Hand in shelf took a genuine product
            if (hand_in_shelf or p.hand_in_shelf_counter >= 2) and object_in_hand and not p.holding_item:
                p.pick_candidate_frames += 1
                if p.pick_candidate_frames >= 2:
                    p.holding_item = True
                    p.item_took_time = t
                    p.item_color_bgr = obj_color or (0, 200, 255)
                    p.item_box = detected_obj_box
                    p.last_item_seen_time = t
                    p.state = "HOLDING PRODUCT"
                    p.held_items_count += 1
                    counts["took_product"] += 1
                    counts["currently_held"] += 1
                    log_event(t, f"Customer #{p.id}: TOOK PRODUCT from shelf (Holding: {counts['currently_held']})", AMB)
                    p.pick_candidate_frames = 0
            else:
                p.pick_candidate_frames = 0

            # 2. While holding item: check whether kept, returned, placed in bag, or stolen
            if p.holding_item:
                if object_in_hand:
                    p.last_item_seen_time = t
                    p.item_box = detected_obj_box
                    p.state = "HOLDING PRODUCT"
                    p.return_candidate_frames = 0
                    p.bag_drop_candidate_frames = 0
                    p.disappear_candidate_frames = 0
                else:
                    # Item not currently observed in hand: evaluate Destination
                    # Destination A: Placed into Shoulder Bag or Cover
                    hand_at_bag = False
                    matched_bag_label = p.bag_type

                    if p.bag_box is not None:
                        bag_cx = (p.bag_box[0] + p.bag_box[2]) / 2.0
                        bag_cy = (p.bag_box[1] + p.bag_box[3]) / 2.0
                        for h_idx in (9, 10):
                            w_pt = k[h_idx][:2]
                            dist_to_bag = np.hypot(w_pt[0] - bag_cx, w_pt[1] - bag_cy)
                            if point_in_box(w_pt, p.bag_box, margin=int(0.25 * ph)) or dist_to_bag < 0.28 * ph:
                                hand_at_bag = True
                                break

                    if not hand_at_bag:
                        for c in containers_det:
                            cb = c["box"]
                            for h_idx in (9, 10):
                                w_pt = k[h_idx][:2]
                                if point_in_box(w_pt, cb, margin=int(0.22 * ph)):
                                    hand_at_bag = True
                                    matched_bag_label = c.get("label", "COVER / BAG")
                                    break
                            if hand_at_bag:
                                break

                    # Destination B: Returned to Shelf
                    hand_back_on_shelf = any(
                        cv2.pointPolygonTest(s, (float(k[h_idx][0]), float(k[h_idx][1])), True) >= -reach_margin
                        for s in shelf_np
                        for h_idx in (9, 10)
                        if k[h_idx][2] > 0.15
                    )

                    if hand_at_bag:
                        p.bag_drop_candidate_frames += 1
                        if p.bag_drop_candidate_frames >= 2:
                            p.holding_item = False
                            p.held_items_count = max(0, p.held_items_count - 1)
                            p.state = "PLACED IN COVER (SAFE)"
                            counts["placed_cart"] += 1
                            counts["currently_held"] = max(0, counts["currently_held"] - 1)
                            log_event(t, f"Customer #{p.id}: Product placed into {matched_bag_label} (Total in Bag: {counts['placed_cart']})", GRN)
                            p.bag_drop_candidate_frames = 0
                            p.return_candidate_frames = 0
                            p.disappear_candidate_frames = 0

                    elif hand_back_on_shelf:
                        p.return_candidate_frames += 1
                        if p.return_candidate_frames >= 2:
                            p.holding_item = False
                            p.held_items_count = max(0, p.held_items_count - 1)
                            p.state = "PUT BACK ON SHELF"
                            counts["put_back"] += 1
                            counts["currently_held"] = max(0, counts["currently_held"] - 1)
                            log_event(t, f"Customer #{p.id}: Product returned to shelf (Total Returned: {counts['put_back']})", CYN)
                            p.bag_drop_candidate_frames = 0
                            p.return_candidate_frames = 0
                            p.disappear_candidate_frames = 0

                    else:
                        # Neither returned nor placed in bag: product is missing / concealed!
                        time_since_disappear = t - p.last_item_seen_time
                        p.disappear_candidate_frames += 1
                        if time_since_disappear >= 0.35 and p.disappear_candidate_frames >= 4:
                            p.holding_item = False
                            p.held_items_count = max(0, p.held_items_count - 1)
                            p.theft_flagged = True
                            p.state = "THEFT ALERT (ITEM MISSING)"
                            counts["theft_suspicion"] += 1
                            counts["currently_held"] = max(0, counts["currently_held"] - 1)
                            log_event(t, f"Customer #{p.id}: 🚨 THEFT ALERT! Product missing/concealed (Not in hand, Not in bag, Not returned)", RED)
                            p.bag_drop_candidate_frames = 0
                            p.return_candidate_frames = 0
                            p.disappear_candidate_frames = 0

            else:
                # Default state if not holding item
                if hand_in_shelf:
                    p.state = "REACHING SHELF"
                elif not p.theft_flagged:
                    p.state = "BROWSING"

        # ────────────────────────── RENDER HUD ──────────────────────────
        canvas = np.zeros((720, total_w, 3), np.uint8)
        canvas[:] = DARK

        # Resized main video view
        canvas[:, :vw] = cv2.resize(frame, (vw, 720))

        # Draw Shelf Polygons
        for s in shelf_np:
            pts = np.stack([s[:, 0] * sx_render, s[:, 1] * sy_render], axis=1).astype(np.int32)
            ov = canvas.copy()
            cv2.fillPoly(ov, [pts], (0, 180, 220))
            cv2.addWeighted(ov, 0.18, canvas, 0.82, 0, canvas)
            cv2.polylines(canvas, [pts], True, (0, 240, 240), 2, cv2.LINE_AA)
            cx_ = int(pts[:, 0].mean())
            cy_ = int(pts[:, 1].mean())
            cv2.putText(canvas, "SHELF ROI", (cx_ - 38, cy_), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 240), 2, cv2.LINE_AA)

        # Draw Visual Elements for Customers
        for p in active:
            b = p.d["box"]
            k = p.d["k"]
            b_sc = [int(b[0] * sx_render), int(b[1] * sy_render), int(b[2] * sx_render), int(b[3] * sy_render)]

            # Customer bounding box color based on state
            box_col = RED if "THEFT" in p.state else (GRN if "SAFE" in p.state else (AMB if "HOLDING" in p.state else (CYN if "REACHING" in p.state else (160, 160, 160))))
            cv2.rectangle(canvas, (b_sc[0], b_sc[1]), (b_sc[2], b_sc[3]), box_col, 2)
            cv2.putText(canvas, f"#{p.id} {p.state}", (b_sc[0], max(18, b_sc[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_col, 2)

            # Draw tracked Shoulder Bag / Bag
            if p.bag_box is not None:
                bx1 = int(p.bag_box[0] * sx_render)
                by1 = int(p.bag_box[1] * sy_render)
                bx2 = int(p.bag_box[2] * sx_render)
                by2 = int(p.bag_box[3] * sy_render)

                bag_ov = canvas.copy()
                cv2.rectangle(bag_ov, (bx1, by1), (bx2, by2), (0, 200, 100), -1)
                cv2.addWeighted(bag_ov, 0.20, canvas, 0.80, 0, canvas)
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (50, 220, 120), 2)
                cv2.putText(canvas, f"[{p.bag_type}]", (bx1, max(14, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 220, 120), 1, cv2.LINE_AA)

            # Draw Hands & Held Product
            for w_idx in (9, 10):
                w = k[w_idx]
                if w[2] > 0.15:
                    wx = int(w[0] * sx_render)
                    wy = int(w[1] * sy_render)
                    hand_col = AMB if p.holding_item else (CYN if p.hands_in_shelf else GRN)
                    cv2.circle(canvas, (wx, wy), 6, hand_col, -1)
                    cv2.circle(canvas, (wx, wy), 9, (255, 255, 255), 1)

            if p.holding_item and p.item_box is not None:
                ix1 = int(p.item_box[0] * sx_render)
                iy1 = int(p.item_box[1] * sy_render)
                ix2 = int(p.item_box[2] * sx_render)
                iy2 = int(p.item_box[3] * sy_render)
                cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (0, 215, 255), 2)
                cv2.putText(canvas, "PRODUCT", (ix1, max(14, iy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

        # ────────────────────────── TELEMETRY PANEL ──────────────────────────
        px0 = vw
        panel = canvas[:, px0:total_w]
        panel[:] = (20, 20, 24)

        # Header
        cv2.putText(panel, "RETAIL SECURITY AI", (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.70, WHT, 2)
        now = time.time()
        fps_disp = 0.88 * fps_disp + 0.12 * (1.0 / max(1e-4, now - t_prev))
        cv2.putText(panel, f"{t:5.1f}s | {fps_disp:4.1f} fps", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # Dynamic Status Banner
        any_theft = any(p.theft_flagged for p in people.values())
        any_holding = any(p.holding_item for p in people.values())
        any_reaching = any(p.hands_in_shelf for p in people.values())
        any_safe = any("SAFE" in p.state for p in people.values())

        if any_theft:
            banner_col, banner_text = RED, "🚨 THEFT DETECTED"
        elif any_holding:
            banner_col, banner_text = AMB, "🛍 HOLDING PRODUCT"
        elif any_safe:
            banner_col, banner_text = GRN, "👜 PLACED IN BAG (SAFE)"
        elif any_reaching:
            banner_col, banner_text = CYN, "✋ REACHING SHELF"
        else:
            banner_col, banner_text = GRN, "? BROWSING"

        cv2.rectangle(panel, (15, 75), (PW - 15, 120), banner_col, -1)
        cv2.putText(panel, banner_text, (25, 106), cv2.FONT_HERSHEY_DUPLEX, 0.65, WHT, 2)

        # Live Metric Counts
        y_c = 155
        dy_c = 28
        metrics = [
            ("Customers in view", len(active), WHT),
            ("Took Product", counts["took_product"], AMB),
            ("Currently in Hand", counts["currently_held"], (0, 215, 255)),
            ("Placed in Cover/Bag", counts["placed_cart"], GRN),
            ("Put Back on Shelf", counts["put_back"], CYN),
            ("THEFT ALERTS", counts["theft_suspicion"], RED),
        ]
        for label, val, c_col in metrics:
            cv2.putText(panel, label, (20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
            cv2.putText(panel, str(val), (PW - 45, y_c), cv2.FONT_HERSHEY_DUPLEX, 0.58, c_col, 2)
            y_c += dy_c

        # Activity Log Divider
        cv2.line(panel, (15, y_c + 8), (PW - 15, y_c + 8), (55, 55, 65), 1)
        y_c += 28
        cv2.putText(panel, "— ACTIVITY LOG —", (20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 140), 1)
        y_c += 22

        # Display last 10 log messages
        for ev_t, ev_text, ev_col in events[-10:]:
            short_txt = (ev_text[:38] + "..") if len(ev_text) > 40 else ev_text
            cv2.putText(panel, f"{ev_t:4.1f}s {short_txt}", (20, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.40, ev_col, 1)
            y_c += 20

        # Footer
        cv2.putText(panel, "Shelf Tracking • Shoulder Bag • Anti-Theft Guard", (18, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 100), 1)

        # Save and display
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
    for k, v in counts.items():
        print(f"  {k:<22}: {v}")
    print("=" * 64)
    return events, counts


# ─────────────────────────── CLI ENTRYPOINT ─────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Retail Theft Detection v2")
    parser.add_argument("video", nargs="?", default="Video Project 1.mp4", help="Video path")
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
