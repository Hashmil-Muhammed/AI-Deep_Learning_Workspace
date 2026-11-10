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
    def __init__(self, model_det="yolov8m.pt", model_pose="yolov8m-pose.pt"):
        self.device = get_device()
        print(f"[Detector] Initializing using device: {self.device.upper()}")

        # Check if .pt weights exist, else check .onnx
        pt_det = os.path.join(HERE, model_det)
        pt_pose = os.path.join(HERE, model_pose)

        self.use_pt = HAS_ULTRALYTICS and os.path.exists(pt_det) and os.path.exists(pt_pose)

        if self.use_pt:
            print(f"[Detector] Loading PyTorch models: {model_det}, {model_pose}")
            self.model_det = YOLO(pt_det)
            self.model_pose = YOLO(pt_pose)
        else:
            print("[Detector] PyTorch weights not found or ultralytics not installed. Loading ONNX models...")
            so = ort.SessionOptions()
            so.intra_op_num_threads = min(4, os.cpu_count() or 2)
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            self.sess_det = ort.InferenceSession(os.path.join(HERE, "models", "yolov8m.onnx"), so, providers=providers)
            self.sess_pose = ort.InferenceSession(os.path.join(HERE, "models", "yolov8m-pose.onnx"), so, providers=providers)

    def detect_objects_and_pose(self, frame, conf_det=0.25, conf_pose=0.25):
        """
        Returns:
            persons: list of dicts(box=[x1,y1,x2,y2], conf, k=17x3 keypoints)
            containers: list of dicts(box=[x1,y1,x2,y2], cls, conf) (backpacks, bags, carts)
        """
        if self.use_pt:
            # 1. Pose estimation
            pose_res = self.model_pose.predict(frame, conf=conf_pose, device=self.device, verbose=False)[0]
            persons = []
            if pose_res.boxes is not None and len(pose_res.boxes) > 0:
                boxes = pose_res.boxes.xyxy.cpu().numpy()
                confs = pose_res.boxes.conf.cpu().numpy()
                kpts = pose_res.keypoints.data.cpu().numpy() # [N, 17, 3]
                for b, c, k in zip(boxes, confs, kpts):
                    persons.append(dict(box=b.tolist(), conf=float(c), k=k))

            # 2. General object detection for carts / bags
            det_res = self.model_det.predict(frame, conf=conf_det, device=self.device, verbose=False)[0]
            containers = []
            if det_res.boxes is not None and len(det_res.boxes) > 0:
                d_boxes = det_res.boxes.xyxy.cpu().numpy()
                d_confs = det_res.boxes.conf.cpu().numpy()
                d_cls = det_res.boxes.cls.cpu().numpy().astype(int)
                # COCO classes: 24: handbag, 26: handbag/shopping bag, 28: suitcase, 73: book, etc.
                for b, c, cl in zip(d_boxes, d_confs, d_cls):
                    if cl in (24, 26, 28):  # bags, baskets, luggage
                        containers.append(dict(box=b.tolist(), conf=float(c), cls=cl))
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

    # 3. Vivid / saturated colors (packaging, cans, boxes, products)
    vivid_mask = cv2.inRange(hsv, (0, 60, 50), (180, 255, 255))

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

    # Significant product area in hand (>= 5% of hand padding window)
    if area >= 20 and ratio >= 0.05:
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
        self.state = "BROWSING"   # BROWSING, TOOK PRODUCT, PLACED IN CART, THEFT SUSPICION
        self.hands_in_shelf = False
        self.holding_item = False
        self.item_color_bgr = None
        self.item_box = None
        self.item_took_time = None
        self.last_item_seen_time = -99.0
        self.theft_flagged = False
        self.history = collections.deque(maxlen=30)


# ─────────────────────────── MAIN DETECTION LOOP ────────────────────────────
def run_theft_detection(video_path, shelf_polygons, save=True, show=True):
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

    # Initialize Detector
    detector = DetectorWrapper()

    # Video Writer
    out_file = os.path.join(HERE, "out_theft_v2.mp4")
    wr = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, 720)) if save else None

    if show:
        cv2.namedWindow("AI Retail Theft Detection v2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Retail Theft Detection v2", total_w, 720)

    people = {}
    nid = 1
    events = []
    counts = dict(took_product=0, placed_cart=0, put_back=0, theft_suspicion=0)

    def log_event(t, text, col):
        events.append((t, text, col))
        prefix = "🚨 " if col is RED else ("🛍 " if col is AMB else "✓ ")
        print(f"[{t:5.1f}s] {prefix}{text}")

    idx = 0
    t_prev = time.time()
    fps_disp = 0.0

    print("\n[Pipeline] Detection running... (Press 'Q' to stop)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        t = idx / fps

        # 1. Run YOLOv8m (person & cart/bag detection) and YOLOv8m-pose
        persons_det, containers_det = detector.detect_objects_and_pose(frame, conf_det=0.25, conf_pose=0.25)

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

            # Check hands inside shelf ROI
            hand_in_shelf = False
            object_in_hand = False
            detected_obj_box = None
            obj_color = None

            for h_idx in (9, 10):  # Left & Right Wrists
                w = k[h_idx]
                if w[2] < 0.20:
                    continue
                w_pt = (int(w[0]), int(w[1]))

                # Check if wrist is inside or touching shelf ROI
                in_shelf = any(point_in_poly(w_pt, s) for s in shelf_np)
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

            p.hands_in_shelf = hand_in_shelf

            # ──────────────────── STATE MACHINE ────────────────────
            # Condition A: Person taking product from shelf
            if hand_in_shelf and object_in_hand and not p.holding_item:
                # Check shelf ROI change: has shelf background changed at this pick location?
                diff_shelf = cv2.absdiff(frame, clean_shelf_bg)
                diff_val = cv2.mean(diff_shelf, mask=shelf_mask)[0]

                p.holding_item = True
                p.item_took_time = t
                p.item_color_bgr = obj_color
                p.item_box = detected_obj_box
                p.last_item_seen_time = t
                p.state = "TOOK PRODUCT"
                counts["took_product"] += 1
                log_event(t, f"Customer #{p.id}: TOOK PRODUCT from shelf", AMB)

            elif p.holding_item:
                if object_in_hand:
                    p.last_item_seen_time = t
                    p.item_box = detected_obj_box
                    p.state = "TOOK PRODUCT"
                else:
                    # Item disappeared from hand! Check WHERE it disappeared to:
                    time_since_disappear = t - p.last_item_seen_time
                    if time_since_disappear >= 0.40:
                        # 1. Did it go into a Cart / Basket / Bag?
                        in_cart = False
                        for c in containers_det:
                            c_box = c["box"]
                            for h_idx in (9, 10):
                                w_pt = k[h_idx][:2]
                                if point_in_box(w_pt, c_box, margin=int(0.10 * ph)):
                                    in_cart = True
                                    break
                            if in_cart:
                                break

                        # 2. Was it put back on the shelf?
                        hands_back_on_shelf = any(point_in_poly(k[h_idx][:2], s) for s in shelf_np for h_idx in (9, 10) if k[h_idx][2] > 0.20)

                        if in_cart:
                            p.holding_item = False
                            p.state = "PLACED IN CART (SAFE)"
                            counts["placed_cart"] += 1
                            log_event(t, f"Customer #{p.id}: Placed item into cart/basket (SAFE)", GRN)

                        elif hands_back_on_shelf:
                            p.holding_item = False
                            p.state = "PUT BACK ON SHELF"
                            counts["put_back"] += 1
                            log_event(t, f"Customer #{p.id}: Item put back on shelf", GRN)

                        else:
                            # 3. Did it disappear into body (jacket, pocket, waistband)?
                            wrists_near_body = False
                            for h_idx in (9, 10):
                                w_pt = k[h_idx][:2]
                                if point_in_box(w_pt, jacket_torso, margin=int(0.08 * ph)) or \
                                   point_in_box(w_pt, pocket_waist, margin=int(0.08 * ph)):
                                    wrists_near_body = True
                                    break

                            # Also check if hands went completely inside jacket/hidden
                            hands_occluded = (k[9][2] < 0.20 or k[10][2] < 0.20)

                            if wrists_near_body or hands_occluded:
                                p.holding_item = False
                                p.state = "THEFT SUSPICION"
                                p.theft_flagged = True
                                counts["theft_suspicion"] += 1
                                log_event(t, f"Customer #{p.id}: Item concealed into pocket/jacket -> THEFT SUSPICION", RED)
                            else:
                                p.holding_item = False
                                p.state = "BROWSING"

            else:
                # If simply standing or browsing
                if not p.theft_flagged:
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

        # Draw Detected Containers / Carts
        for c in containers_det:
            cb = [int(c["box"][0] * sx_render), int(c["box"][1] * sy_render),
                  int(c["box"][2] * sx_render), int(c["box"][3] * sy_render)]
            cv2.rectangle(canvas, (cb[0], cb[1]), (cb[2], cb[3]), (255, 140, 0), 2)
            cv2.putText(canvas, "CART / BAG", (cb[0], max(16, cb[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 140, 0), 1)

        # Draw Customers
        any_theft = any(p.state == "THEFT SUSPICION" or p.theft_flagged for p in active)

        for p in active:
            b_ = [
                int(p.d["box"][0] * sx_render),
                int(p.d["box"][1] * sy_render),
                int(p.d["box"][2] * sx_render),
                int(p.d["box"][3] * sy_render)
            ]
            is_theft = (p.state == "THEFT SUSPICION" or p.theft_flagged)
            is_took = (p.state == "TOOK PRODUCT")
            is_safe = (p.state == "PLACED IN CART (SAFE)")

            col = RED if is_theft else (AMB if is_took else (GRN if is_safe else (120, 220, 120)))

            # Skeleton
            for a_i, c_i in SKEL_EDGES:
                if p.d["k"][a_i][2] > 0.20 and p.d["k"][c_i][2] > 0.20:
                    pt_a = (int(p.d["k"][a_i][0] * sx_render), int(p.d["k"][a_i][1] * sy_render))
                    pt_b = (int(p.d["k"][c_i][0] * sx_render), int(p.d["k"][c_i][1] * sy_render))
                    cv2.line(canvas, pt_a, pt_b, (220, 180, 80), 1, cv2.LINE_AA)

            # Hand Padding Boxes
            ph = p.d["box"][3] - p.d["box"][1]
            pad_size = max(24, int(0.12 * ph))
            for h_idx in (9, 10):
                w = p.d["k"][h_idx]
                if w[2] > 0.20:
                    wx, wy = int(w[0] * sx_render), int(w[1] * sy_render)
                    pad_disp = int(pad_size * sx_render)
                    cv2.rectangle(canvas, (wx - pad_disp, wy - pad_disp), (wx + pad_disp, wy + pad_disp), (0, 255, 255), 1)
                    cv2.circle(canvas, (wx, wy), 4, (0, 255, 0), -1)

            # Person Bounding Box & Status Badge
            cv2.rectangle(canvas, (b_[0], b_[1]), (b_[2], b_[3]), col, 3 if is_theft else 2)

            lbl = f"Customer #{p.id} [{p.state}]"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            cv2.rectangle(canvas, (b_[0], b_[1] - th - 8), (b_[0] + tw + 6, b_[1]), col, -1)
            cv2.putText(canvas, lbl, (b_[0] + 3, b_[1] - 5), cv2.FONT_HERSHEY_DUPLEX, 0.52, WHT if is_theft else BLK, 1, cv2.LINE_AA)

            # Body Zones Highlight when took product
            if is_took or is_theft:
                j_torso, p_waist = get_body_zones(p.d)
                for zone, zcol in ((j_torso, (80, 160, 255)), (p_waist, (40, 40, 230))):
                    zp = np.array([
                        [zone[0] * sx_render, zone[1] * sy_render],
                        [zone[2] * sx_render, zone[1] * sy_render],
                        [zone[2] * sx_render, zone[3] * sy_render],
                        [zone[0] * sx_render, zone[3] * sy_render]
                    ], np.int32)
                    ov2 = canvas.copy()
                    cv2.fillPoly(ov2, [zp], zcol)
                    cv2.addWeighted(ov2, 0.16, canvas, 0.84, 0, canvas)
                    cv2.polylines(canvas, [zp], True, zcol, 1, cv2.LINE_AA)

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
        active_theft = any(p.state == "THEFT SUSPICION" or p.theft_flagged for p in active)
        active_took = any(p.state == "TOOK PRODUCT" for p in active)

        if active_theft:
            bc, bt = RED, "🚨 THEFT SUSPICION"
        elif active_took:
            bc, bt = AMB, "🛍 TOOK PRODUCT"
        else:
            bc, bt = (40, 130, 40), "👀 BROWSING"

        cv2.rectangle(canvas, (x0p - 4, 70), (xrp + 4, 122), bc, -1)
        put_text(bt, 108, WHT, 0.76, 2)

        stat_rows = [
            ("Customers in view",   len(active),               WHT),
            ("Took Product",        counts["took_product"],     AMB),
            ("Placed in Cart/Bag",  counts["placed_cart"],      GRN),
            ("Put Back on Shelf",   counts["put_back"],         CYN),
            ("THEFT SUSPICIONS",    counts["theft_suspicion"],  RED),
        ]
        for i, (lbl, val, col) in enumerate(stat_rows):
            put_text(lbl,      144 + 30 * i, (200, 200, 200), 0.46)
            put_text(str(val), 144 + 30 * i, col, 0.70, 2, right=True)

        put_text("── ACTIVITY LOG ──", 336, (120, 120, 120), 0.42)
        for i, (et, etxt, ecol) in enumerate(events[-9:]):
            put_text(f"{et:5.1f}s {etxt}"[:42], 360 + 28 * i, ecol, 0.36)

        put_text("YOLOv8m + Pose + Hand Padding Segmentation", 710, (80, 80, 80), 0.33)

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
    parser = argparse.ArgumentParser(description="AI Retail Theft Detection v2")
    parser.add_argument("video", nargs="?", default="Video Project 1.mp4", help="Video path")
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
        show=not args.headless
    )

if __name__ == "__main__":
    main()
