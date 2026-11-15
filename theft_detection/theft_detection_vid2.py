"""
theft_detection_v2.py
====================
Theft Detection Pipeline based on User Requirements:
1. Interactive Shelf ROI Drawer:
   - User draws ROI polygon for the shelf.
   - Initial clean shelf ROI is captured and maintained as the reference background.
2. Person & Cart/Basket Detection:
   - Uses YOLOv8m (PyTorch / ONNX with auto GPU/CPU fallback) to detect person, cart, backpack, handbag, basket.
3. Pose Estimation:
   - Uses YOLOv8m-pose to detect 17 person keypoints, focusing on left & right hands/wrists.
4. Hand Padding & Non-Skin Color/Object Detection:
   - Creates a configurable padding box around the person's hands.
   - Isolates skin colors and background shelf pixels to check if anything of a foreign/object color is in hand.
5. Shelf Interaction & State Machine:
   - If person is simply standing there looking/reaching without taking anything -> State: 'BROWSING'.
   - If person's hand enters shelf ROI, takes an item, shelf ROI background changes, and an object is verified in hand:
     -> State: 'TOOK PRODUCT'
6. Outcome Classification:
   - If person places the item into a shopping cart/bag/basket -> State: 'PLACED IN CART (SAFE)'.
   - If the item in hand disappears into the person's body (jacket, pocket, waistband, underarm) -> State: 'THEFT SUSPICION'.
   - If the item is returned to the shelf -> State: 'PUT BACK'.
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
        print(f"[Detector] Initializing using device: {self.device.upper()}")
        
        self.use_pt = HAS_ULTRALYTICS
        
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
                raise FileNotFoundError(
                    f"Model files not found! Neither Ultralytics is installed nor do ONNX models exist at '{onnx_det_path}'."
                )
            so = ort.SessionOptions()
            so.intra_op_num_threads = min(4, os.cpu_count() or 2)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            self.sess_det = ort.InferenceSession(onnx_det_path, so, providers=providers)
            self.sess_pose = ort.InferenceSession(onnx_pose_path, so, providers=providers)

    def detect_objects_and_pose(self, frame, conf_det=0.25, conf_pose=0.25):
        """
        Returns:
            persons: list of dicts(box=[x1,y1,x2,y2], conf, k=17x3 keypoints)
            containers: list of dicts(box=[x1,y1,x2,y2], cls, conf) (backpacks, bags, carts)
        """
        if self.use_pt:
            # 1. Pose estimation (runs inference at imgsz for high speed)
            pose_res = self.model_pose.predict(frame, conf=conf_pose, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            persons = []
            if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                boxes = pose_res.boxes.xyxy.cpu().numpy()
                confs = pose_res.boxes.conf.cpu().numpy()
                kpts = pose_res.keypoints.data.cpu().numpy() # [N, 17, 3]
                for b, c, k in zip(boxes, confs, kpts):
                    persons.append(dict(box=b.tolist(), conf=float(c), k=k))

            # 2. General object detection for covers, bags, backpacks, bottles, items
            det_res = self.model_det.predict(frame, conf=conf_det, imgsz=self.imgsz, device=self.device, verbose=False)[0]
            containers = []
            if det_res.boxes is not None and len(det_res.boxes) > 0:
                d_boxes = det_res.boxes.xyxy.cpu().numpy()
                d_confs = det_res.boxes.conf.cpu().numpy()
                d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
                # COCO classes: 24: backpack, 26: handbag, 28: suitcase, 39: bottle, 73: book, etc.
                CONTAINER_CLASSES = {24: "BAG / BACKPACK", 26: "HANDBAG / COVER", 28: "SUITCASE / BASKET"}
                for b, c, cl in zip(d_boxes, d_confs, d_cls):
                    if cl in CONTAINER_CLASSES:
                        containers.append(dict(box=b.tolist(), conf=float(c), cls=cl, label=CONTAINER_CLASSES[cl]))
            return persons, containers
        else:
            # ONNX fallback
            return self._infer_onnx(frame, conf_det, conf_pose)

    def _infer_onnx(self, frame, conf_det, conf_pose):
        h, w = frame.shape[:2]
        input_sz = 640
        r = input_sz / max(h, w)
        nh, nw = int(h * r), int(w * r)
        can = np.full((input_sz, input_sz, 3), 114, np.uint8)
        can[:nh, :nw] = cv2.resize(frame, (nw, nh))
        blob = can[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        # Run Pose
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

        # Run Det
        out_det = self.sess_det.run(None, {"images": blob})[0][0].T
        sc_d = out_det[:, 4:].max(1)
        mask_d = sc_d > conf_det
        out_d, sc_d = out_det[mask_d], sc_d[mask_d]
        containers = []
        if len(out_d):
            bx_d = np.stack([out_d[:, 0] - out_d[:, 2] / 2, out_d[:, 1] - out_d[:, 3] / 2, out_d[:, 2], out_d[:, 3]], 1) / r
            cls_d = out_d[:, 4:].argmax(1)
            keep_d = np.array(cv2.dnn.NMSBoxes(bx_d.tolist(), sc_d.tolist(), conf_det, 0.45)).flatten()
            for i in keep_d:
                c_id = int(cls_d[i])
                if c_id in (24, 26, 28):
                    x, y, bw, bh = bx_d[i]
                    containers.append(dict(box=[x, y, x + bw, y + bh], conf=float(sc_d[i]), cls=c_id))

        return persons, containers


# ────────────────────────── COLOR & SKIN DETECTION ─────────────────────────
def is_skin_pixel(hsv_pixel, ycrcb_pixel):
    """
    Standard robust human skin color filter combining HSV and YCrCb color spaces.
    Skin tones cluster within:
      HSV: H in [0, 25], S in [30, 200], V in [60, 255]
      YCrCb: Cr in [133, 173], Cb in [77, 127]
    """
    h, s, v = hsv_pixel
    y, cr, cb = ycrcb_pixel
    hsv_skin = (0 <= h <= 25 or 170 <= h <= 180) and (25 <= s <= 190) and (50 <= v <= 255)
    ycrcb_skin = (133 <= cr <= 173) and (77 <= cb <= 127)
    return hsv_skin and ycrcb_skin

def analyze_hand_patch(img, wrist_pt, padding_size=32, bg_patch=None):
    """
    Creates a padding box around the person's hand.
    Analyzes whether anything of a color OTHER THAN hand/skin is present.
    Returns:
        has_object: bool
        object_mask: binary mask of non-skin foreign object
        object_box: (x1, y1, x2, y2)
        non_skin_ratio: float
        dominant_color_bgr: tuple
    """
    H, W = img.shape[:2]
    x, y = int(wrist_pt[0]), int(wrist_pt[1])
    x1 = max(0, x - padding_size)
    y1 = max(0, y - padding_size)
    x2 = min(W, x + padding_size)
    y2 = min(H, y + padding_size)

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return False, None, None, 0.0, None

    patch = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

    # 1. Detect Skin in the hand region
    hsv_skin_mask = cv2.inRange(hsv, (0, 25, 50), (25, 190, 255)) | cv2.inRange(hsv, (170, 25, 50), (180, 190, 255))
    ycrcb_skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin_mask = cv2.bitwise_and(hsv_skin_mask, ycrcb_skin_mask)

    # 2. Difference from shelf background if available
    diff_mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    if bg_patch is not None and bg_patch.shape == patch.shape:
        diff = cv2.absdiff(patch, bg_patch)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_mask = (diff_gray > 25).astype(np.uint8) * 255

    # 3. Vivid / saturated colors (packaging, cans, boxes, products) or high contrast
    vivid_mask = cv2.inRange(hsv, (0, 50, 40), (180, 255, 255))

    # Combine cues for object: (vivid OR changed from background) AND NOT skin
    candidate = cv2.bitwise_or(vivid_mask, diff_mask) if bg_patch is not None else vivid_mask
    object_mask = cv2.bitwise_and(candidate, cv2.bitwise_not(skin_mask))
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    cnts, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, None, None, 0.0, None

    # Find the largest non-skin blob
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_area = (x2 - x1) * (y2 - y1)
    ratio = area / total_area

    # Product area in hand (>= 3.5% of hand padding window or area >= 15px)
    if area >= 15 and ratio >= 0.035:
        bx, by, bw, bh = cv2.boundingRect(c)
        global_box = [x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh]
        
        # Mean color of detected object
        mean_bgr = cv2.mean(patch, mask=object_mask)[:3]
        return True, object_mask, global_box, ratio, mean_bgr

    return False, None, None, ratio, None


# ─────────────────────────── GEOMETRY & ZONES ───────────────────────────────
def get_body_zones(d):
    """
    Computes body zones from pose keypoints:
      - Torso / Jacket Inside Zone: between shoulders and hips
      - Pocket / Waistband Zone: hips to upper thighs
    """
    k, b = d["k"], d["box"]
    bh = b[3] - b[1]
    sh = [q for q in k[[5, 6]] if q[2] > 0.20]
    hp = [q for q in k[[11, 12]] if q[2] > 0.20]
    sy = float(np.mean([q[1] for q in sh])) if sh else b[1] + 0.18 * bh
    hy = float(np.mean([q[1] for q in hp])) if hp else sy + 0.42 * bh
    xs = [q[0] for q in sh + hp]
    cx = float(np.mean(xs)) if xs else (b[0] + b[2]) / 2.0
    L = max(hy - sy, 0.20 * bh)
    hw = max((abs(sh[0][0] - sh[1][0]) / 2.0 * 1.35 if len(sh) == 2 else 0), 0.24 * L) + 0.32 * L

    jacket_torso = [cx - hw, sy - 0.05 * L, cx + hw, sy + 0.65 * L]
    pocket_waist = [cx - hw, sy + 0.55 * L, cx + hw, hy + 0.60 * L]
    return jacket_torso, pocket_waist

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0

def point_in_box(pt, box, margin=0):
    return (box[0] - margin <= pt[0] <= box[2] + margin and
            box[1] - margin <= pt[1] <= box[3] + margin)

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


# ─────────────────────────── INTERACTIVE ROI DRAWER ─────────────────────────
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

    def mouse_callback(event, x, y, flags, _):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            cur.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(cur) >= 3:
                polys.append(list(cur))
                cur.clear()
                print(f"Shelf ROI #{len(polys)} captured.")
            else:
                print("Click at least 3 points to complete a shelf polygon.")

    win = "Draw Shelf ROI  |  Left-Click: Add Point  |  Right-Click: Close Shelf  |  Enter: Start"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    cv2.setMouseCallback(win, mouse_callback)

    print("\n" + "="*65)
    print(" STEP 1: DRAW SHELF ROI ON SCREEN")
    print(" • Left-Click  : Place corner of shelf")
    print(" • Right-Click : Close current shelf polygon")
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

        cv2.putText(vis, f"Shelves defined: {len(polys)} | Current points: {len(cur)}", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
        cv2.putText(vis, "Left-Click: Add | Right-Click: Close | Enter: Run | Z: Undo | R: Reset", (14, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1)

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('z'), ord('Z')):
            if cur: cur.pop()
        elif key in (ord('r'), ord('R')):
            polys.clear(); cur.clear()
        elif key in (13, ord('s'), ord('S'), 32):
            if len(cur) >= 3:
                polys.append(list(cur))
                cur.clear()
            if polys:
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
        self.state = "BROWSING"   # BROWSING, HOLDING PRODUCT, PLACED IN COVER (SAFE), PUT BACK ON SHELF, THEFT ALERT
        self.hands_in_shelf = False
        self.holding_item = False
        self.item_color_bgr = None
        self.item_box = None
        self.item_took_time = None
        self.last_item_seen_time = -99.0
        self.hand_in_shelf_counter = 0  # smoothing frames in shelf
        self.hand_away_counter = 0      # smoothing frames hand leaving shelf with product
        self.theft_flagged = False
        self.bag_box = None             # Tracked bag / cover held by customer
        self.history = collections.deque(maxlen=30)


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

    # Initialize Detector with selected size (nano 'n' runs fast on CPU, 'm' is heavy)
    model_det = f"yolov8{model_size}.pt"
    model_pose = f"yolov8{model_size}-pose.pt"
    detector = DetectorWrapper(model_det=model_det, model_pose=model_pose, imgsz=480)

    # Video Writer
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

    # Cached detections for frame-skipping / temporal smoothness
    cached_persons = []
    cached_containers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        t = idx / fps

        # 1. Run YOLO detection every (skip_frames + 1) frames to keep CPU smooth
        if idx % (skip_frames + 1) == 1 or not cached_persons:
            persons_det, containers_det = detector.detect_objects_and_pose(frame, conf_det=0.25, conf_pose=0.25)
            cached_persons = persons_det
            cached_containers = containers_det
        else:
            persons_det = cached_persons
            containers_det = cached_containers

        # 2. Tracking Persons across frames
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
            pad_size = max(24, int(0.12 * ph))

            jacket_torso, pocket_waist = get_body_zones(p.d)

            # --- Universal Bag / Cover Detection for Customer ---
            # 1. Prioritize YOLO-detected Handbag / Backpack / Bag on customer
            bag_found = False
            best_c_box = None
            best_c_area = 0

            for c in containers_det:
                cb = c["box"]
                # Must be associated with this customer
                if point_in_box(((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2), b, margin=int(0.20 * ph)):
                    area_c = (cb[2] - cb[0]) * (cb[3] - cb[1])
                    if area_c > best_c_area:
                        best_c_area = area_c
                        best_c_box = cb

            if best_c_box is not None:
                new_bag = [int(best_c_box[0]), int(best_c_box[1]), int(best_c_box[2]), int(best_c_box[3])]
                bag_found = True
            else:
                # 2. Handheld Bag search around holding hand (left/right wrist or hip)
                lw = k[9]
                rw = k[10]
                lh = k[11]
                rh = k[12]

                # Check which wrist is lower (holding bag hanging down)
                hand_cand = lw if (lw[2] > 0.20 and lw[1] > lh[1]) else (rw if (rw[2] > 0.20 and rw[1] > rh[1]) else None)
                base_x = int(hand_cand[0]) if hand_cand is not None else int((lh[0] + rh[0]) / 2)
                base_y = int(hand_cand[1]) if hand_cand is not None else int((lh[1] + rh[1]) / 2)

                sw = max(35, int(0.40 * (b[2] - b[0])))
                sh = max(45, int(0.45 * ph))

                sx1 = max(0, base_x - sw // 2)
                sx2 = min(W, base_x + sw // 2)
                sy1 = base_y
                sy2 = min(H, base_y + sh)

                if sy2 > sy1 and sx2 > sx1:
                    sub = frame[sy1:sy2, sx1:sx2]
                    hsv_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
                    # Support white plastic, dark/black leather handbag, and branded bags
                    blue_m = cv2.inRange(hsv_sub, (90, 40, 40), (135, 255, 255))
                    white_m = cv2.inRange(hsv_sub, (0, 0, 105), (180, 80, 255))
                    dark_m = cv2.inRange(hsv_sub, (0, 0, 15), (180, 255, 75))
                    bag_comb = cv2.bitwise_or(blue_m, cv2.bitwise_or(white_m, dark_m))

                    cnts_bag, _ = cv2.findContours(bag_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_bag = [c for c in cnts_bag if cv2.contourArea(c) > 60]

                    if valid_bag:
                        cb = max(valid_bag, key=cv2.contourArea)
                        rx, ry, rw_b, rh_b = cv2.boundingRect(cb)
                        new_bag = [
                            max(0, sx1 + rx - 4),
                            max(0, sy1 + ry - 6),
                            min(W, sx1 + rx + rw_b + 4),
                            min(H, sy1 + ry + rh_b + 6)
                        ]
                    else:
                        new_bag = [sx1, sy1, sx2, sy2]
                    bag_found = True

            if bag_found:
                if p.bag_box is not None:
                    p.bag_box = [
                        int(0.60 * p.bag_box[0] + 0.40 * new_bag[0]),
                        int(0.60 * p.bag_box[1] + 0.40 * new_bag[1]),
                        int(0.60 * p.bag_box[2] + 0.40 * new_bag[2]),
                        int(0.60 * p.bag_box[3] + 0.40 * new_bag[3])
                    ]
                else:
                    p.bag_box = new_bag

            # Check hands inside or reaching into shelf ROI
            hand_in_shelf = False
            object_in_hand = False
            detected_obj_box = None
            obj_color = None

            reach_margin = int(0.08 * ph)  # tolerance margin for reaching shelf edge

            for h_idx in (9, 10):  # Left & Right Wrists
                w = k[h_idx]
                if w[2] < 0.15:
                    continue
                w_pt = (int(w[0]), int(w[1]))

                # Check if wrist is inside or within reach distance of shelf ROI
                in_shelf = any(cv2.pointPolygonTest(s, (float(w_pt[0]), float(w_pt[1])), True) >= -reach_margin for s in shelf_np)
                if in_shelf:
                    hand_in_shelf = True

                # Extract background patch corresponding to this hand location
                x1_bg = max(0, w_pt[0] - pad_size)
                y1_bg = max(0, w_pt[1] - pad_size)
                x2_bg = min(W, w_pt[0] + pad_size)
                y2_bg = min(H, w_pt[1] + pad_size)
                bg_crop = clean_shelf_bg[y1_bg:y2_bg, x1_bg:x2_bg] if (x2_bg > x1_bg and y2_bg > y1_bg) else None

                # Hand Padding analysis: check if any non-skin, non-shelf object is held
                has_obj, o_mask, o_box, o_ratio, o_col = analyze_hand_patch(frame, w_pt, padding_size=pad_size, bg_patch=bg_crop)

                if has_obj:
                    object_in_hand = True
                    detected_obj_box = o_box
                    obj_color = o_col
                elif hand_in_shelf:
                    # When hand is in shelf, create a visible product candidate box around wrist
                    hw = pad_size // 2
                    detected_obj_box = [w_pt[0] - hw, w_pt[1] - hw, w_pt[0] + hw, w_pt[1] + hw]
                    object_in_hand = True
                    obj_color = (0, 200, 255)

            p.hands_in_shelf = hand_in_shelf

            # ──────────────────── STATE MACHINE & PRODUCT TRACKING ────────────────────
            if hand_in_shelf:
                p.hand_in_shelf_counter += 1
                p.hand_away_counter = 0
            else:
                p.hand_in_shelf_counter = max(0, p.hand_in_shelf_counter - 1)
                p.hand_away_counter += 1

            # 1. Product Pick Detection: Hand was in shelf and now holds a product
            if (hand_in_shelf or p.hand_in_shelf_counter >= 2) and (object_in_hand or hand_in_shelf) and not p.holding_item:
                p.holding_item = True
                p.item_took_time = t
                p.item_color_bgr = obj_color or (0, 200, 255)
                p.item_box = detected_obj_box
                p.last_item_seen_time = t
                p.state = "HOLDING PRODUCT"
                counts["took_product"] += 1
                counts["currently_held"] += 1
                log_event(t, f"Customer #{p.id}: TOOK PRODUCT from shelf (Holding: {counts['currently_held']})", AMB)

            elif p.holding_item:
                if object_in_hand:
                    p.last_item_seen_time = t
                    p.item_box = detected_obj_box
                    p.state = "HOLDING PRODUCT"
                else:
                    # Item disappeared from hand! Evaluate Destination (Cover vs Return vs Theft)
                    time_since_disappear = t - p.last_item_seen_time

                    # Destination A: Placed into Bag / Cover
                    in_cover_or_cart = False
                    matched_cover_label = "COVER / BAG"

                    if p.bag_box is not None:
                        bag_cx = (p.bag_box[0] + p.bag_box[2]) / 2.0
                        bag_cy = (p.bag_box[1] + p.bag_box[3]) / 2.0
                        for h_idx in (9, 10):
                            w_pt = k[h_idx][:2]
                            dist_to_bag = np.hypot(w_pt[0] - bag_cx, w_pt[1] - bag_cy)
                            if point_in_box(w_pt, p.bag_box, margin=int(0.28 * ph)) or dist_to_bag < 0.35 * ph:
                                in_cover_or_cart = True
                                break

                    if not in_cover_or_cart:
                        for c in containers_det:
                            c_box = c["box"]
                            for h_idx in (9, 10):
                                w_pt = k[h_idx][:2]
                                if point_in_box(w_pt, c_box, margin=int(0.25 * ph)):
                                    in_cover_or_cart = True
                                    matched_cover_label = c.get("label", "COVER / BAG")
                                    break
                            if in_cover_or_cart:
                                break

                    # Destination B: Returned back onto Shelf
                    hands_back_on_shelf = any(
                        cv2.pointPolygonTest(s, (float(k[h_idx][0]), float(k[h_idx][1])), True) >= -reach_margin
                        for s in shelf_np
                        for h_idx in (9, 10)
                        if k[h_idx][2] > 0.15
                    )

                    if in_cover_or_cart:
                        # Case 1: Product placed into Bag/Cover
                        p.holding_item = False
                        p.state = "PLACED IN COVER (SAFE)"
                        counts["placed_cart"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        log_event(t, f"Customer #{p.id}: Product placed into {matched_cover_label} (Total in Cover: {counts['placed_cart']})", GRN)

                    elif hands_back_on_shelf:
                        # Case 2: Product returned back to Shelf
                        p.holding_item = False
                        p.state = "PUT BACK ON SHELF"
                        counts["put_back"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        log_event(t, f"Customer #{p.id}: Product returned to shelf (Total Returned: {counts['put_back']})", CYN)

                    elif time_since_disappear >= 0.35:
                        # Case 3: Product taken from shelf, NOT in hand, NOT in cover, NOT returned to shelf!
                        # Product is MISSING / CONCEALED -> THEFT ALERT
                        p.holding_item = False
                        p.state = "THEFT ALERT (ITEM MISSING)"
                        p.theft_flagged = True
                        counts["theft_suspicion"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        log_event(t, f"Customer #{p.id}: 🚨 THEFT ALERT! Product missing (Not in hand, Not in cover, Not returned)", RED)

            else:
                # If simply standing or browsing
                if not p.theft_flagged:
                    p.state = "BROWSING"
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

        # Draw Customers and their held Bag / Cover
        any_theft = any(p.state.startswith("THEFT") or p.theft_flagged for p in active)

        for p in active:
            b_ = [
                int(p.d["box"][0] * sx_render),
                int(p.d["box"][1] * sy_render),
                int(p.d["box"][2] * sx_render),
                int(p.d["box"][3] * sy_render)
            ]
            is_theft = (p.state.startswith("THEFT") or p.theft_flagged)
            is_took = ("PRODUCT" in p.state)
            is_safe = ("PLACED IN COVER" in p.state or "SAFE" in p.state)
            is_returned = ("PUT BACK" in p.state)

            col = RED if is_theft else (AMB if is_took else (GRN if is_safe else (CYN if is_returned else (120, 220, 120))))

            # 1. Customer Bounding Box & Status Badge
            cv2.rectangle(canvas, (b_[0], b_[1]), (b_[2], b_[3]), col, 3 if is_theft else 2)

            lbl = f"Customer #{p.id} [{p.state}]"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            cv2.rectangle(canvas, (b_[0], b_[1] - th - 8), (b_[0] + tw + 6, b_[1]), col, -1)
            cv2.putText(canvas, lbl, (b_[0] + 3, b_[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.52, WHT if is_theft else BLK, 1, cv2.LINE_AA)

            # 2. Customer's Bag / Cover Detection Box
            if p.bag_box is not None:
                bb = [
                    int(p.bag_box[0] * sx_render),
                    int(p.bag_box[1] * sy_render),
                    int(p.bag_box[2] * sx_render),
                    int(p.bag_box[3] * sy_render)
                ]
                cv2.rectangle(canvas, (bb[0], bb[1]), (bb[2], bb[3]), (255, 140, 0), 2)
                cv2.putText(canvas, "COVER / BAG", (bb[0], max(14, bb[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 140, 0), 1, cv2.LINE_AA)

            # 3. Highlight ONLY the Taken Product in Hand
            if p.item_box is not None and is_took:
                ib = [
                    int(p.item_box[0] * sx_render),
                    int(p.item_box[1] * sy_render),
                    int(p.item_box[2] * sx_render),
                    int(p.item_box[3] * sy_render)
                ]
                cv2.rectangle(canvas, (ib[0], ib[1]), (ib[2], ib[3]), (0, 220, 255), 2)
                cv2.putText(canvas, "PRODUCT", (ib[0], max(14, ib[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)

        # Theft Flashing Alarm
        if any_theft and (int(t * 3.5) % 2 == 0):
            cv2.rectangle(canvas, (0, 0), (vw - 1, 719), RED, 6)

        # ────────────────────────── SIDE PANEL ──────────────────────────
        x0p = vw + 14
        xrp = total_w - 14
        F = cv2.FONT_HERSHEY_DUPLEX

        def put_text(txt, y, col=WHT, sc=0.55, bold=1, right=False):
            xx = xrp - cv2.getTextSize(txt, F, sc, bold)[0][0] if right else x0p
            cv2.putText(canvas, txt, (xx, y), F, sc, col, bold, cv2.LINE_AA)

        put_text("RETAIL SECURITY AI", 36, WHT, 0.72, 2)
        put_text(f"{t:5.1f}s | {fps_disp:3.0f} fps", 58, (160, 160, 160), 0.44)

        # Status Badge
        active_theft = any(p.state.startswith("THEFT") or p.theft_flagged for p in active)
        active_took = any("PRODUCT" in p.state for p in active)
        active_safe = any("COVER" in p.state or "SAFE" in p.state for p in active)

        if active_theft:
            bc, bt = RED, "🚨 THEFT ALERT!"
        elif active_took:
            bc, bt = AMB, "🛍 PRODUCT TAKEN"
        elif active_safe:
            bc, bt = GRN, "✓ PLACED IN COVER"
        else:
            bc, bt = (40, 130, 40), "👀 BROWSING"

        cv2.rectangle(canvas, (x0p - 4, 70), (xrp + 4, 122), bc, -1)
        put_text(bt, 108, WHT, 0.72, 2)

        stat_rows = [
            ("Customers in view",   len(active),               WHT),
            ("Took from Shelf",     counts["took_product"],     AMB),
            ("Currently in Hand",   counts["currently_held"],   AMB),
            ("Placed in Cover/Bag", counts["placed_cart"],      GRN),
            ("Returned to Shelf",   counts["put_back"],         CYN),
            ("THEFT ALERTS",        counts["theft_suspicion"],  RED),
        ]
        for i, (lbl, val, col) in enumerate(stat_rows):
            put_text(lbl,      144 + 28 * i, (200, 200, 200), 0.45)
            put_text(str(val), 144 + 28 * i, col, 0.68, 2, right=True)

        put_text("── ACTIVITY LOG ──", 328, (120, 120, 120), 0.42)
        for i, (et, etxt, ecol) in enumerate(events[-8:]):
            put_text(f"{et:5.1f}s {etxt}"[:44], 352 + 26 * i, ecol, 0.35)

        put_text("Shelf Tracking + Cover Det + Theft Guard", 710, (80, 80, 80), 0.33)

        if wr:
            wr.write(canvas)
        if show:
            cv2.imshow("AI Retail Theft Detection v2", canvas)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                print("[Pipeline] Stopped by user.")
                break

        now = time.time()
        fps_disp = (0.9 * fps_disp + 0.1 / max(1e-4, now - t_prev)) if fps_disp else 1.0 / max(1e-4, now - t_prev)
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
    parser = argparse.ArgumentParser(description="AI Retail Theft Detection v2 (vid2.mp4)")
    parser.add_argument("video", nargs="?", default="vid2.mp4", help="Video path")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], help="YOLO model size (n=nano is fastest, m=medium)")
    parser.add_argument("--skip-frames", type=int, default=2, help="Number of frames to skip detection (tracks & caches in-between for smoothness)")
    parser.add_argument("--no-save", action="store_true", help="Do not write output video file")
    parser.add_argument("--headless", action="store_true", help="Run without preview window")
    args = parser.parse_args()

    vpath = args.video if os.path.isabs(args.video) else os.path.join(HERE, args.video)
    if not os.path.exists(vpath):
        print(f"Error: Video file not found: {vpath}")
        sys.exit(1)

    print(f"\nTarget Video : {os.path.basename(vpath)}")
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

