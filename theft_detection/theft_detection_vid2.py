"""
AI Retail Surveillance & Theft Detection Pipeline v2.2
================================================================================
Retail Physical Product Tracking, Hand-Product Separation, and Shelf Return Logic

Key Capabilities Implemented:
1. Physical Product Identification:
   - Detects the physical item removed from the shelf using shelf ROI background differencing.
   - Extracts the actual physical merchandise template, true pixel dimensions, and color appearance.
2. True Hand-Product Separation:
   - Hands and products have distinct visual representations, coordinates, and bounding boxes.
   - Eliminates fake/artificial product boxes around wrists, palms, or skin blobs.
   - When the product is occluded by hands, reports "PRODUCT OCCLUDED / UNCERTAIN" rather than
     drawing a fake box around the wrist.
3. 11-State Product Lifecycle State Machine:
   - ON_SHELF -> PICKUP_CANDIDATE -> PICKED_FROM_SHELF -> IN_HAND -> BEING_INSPECTED
     -> TRACKING_UNCERTAIN -> RETURN_CANDIDATE -> RETURNED_TO_SHELF -> PLACED_IN_BAG
     -> MISSING_SUSPICIOUS -> THEFT_ALERT.
4. Idempotent Event Counting:
   - Confirmed Pickup: Total Picked +1, Currently Held +1 (counted exactly once).
   - Confirmed Return: Returned to Shelf +1, Currently Held -1 (counted exactly once).
5. Evidence-Based Theft Verification:
   - Grace period handling for inspection and occlusion; eliminates premature false theft alarms.
   - Alert banner auto-reverts after 6.0 seconds so it never locks the screen permanently.
6. Multi-Person Tracking:
   - Spatial IoU + normalized centroid association with 60-frame buffer (no duplicate IDs).
================================================================================
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))

# ────────────────────────── COLOR PALETTE & STYLING ──────────────────────────
BG_DARK = (20, 22, 26)
PANEL_BG = (28, 32, 38)
ACCENT_BLUE = (235, 145, 30)
GRN = (70, 200, 90)              # Green: Verified Return / Normal Browsing
RED = (60, 60, 240)              # Red: Theft Alert
AMB = (40, 180, 255)             # Amber: Active Product Picked / Holding
ORANGE = (30, 130, 255)          # Orange: Tracking Uncertain / Missing
TXT_WHITE = (245, 245, 245)
TXT_MUTED = (160, 165, 175)
BORDER_COLOR = (55, 62, 72)
SHELF_COLOR = (245, 180, 20)      # Yellow: Shelf ROI Polygon
HAND_COLOR = (255, 190, 0)       # Cyan-Blue: Customer Hand Bounding Box
PROD_COLOR = (0, 215, 255)       # Gold-Amber: Physical Product Bounding Box


# ─────────────────────── 11-STATE PRODUCT LIFECYCLE ─────────────────────────
class ProductState:
    ON_SHELF = "ON_SHELF"
    PICKUP_CANDIDATE = "PICKUP_CANDIDATE"
    PICKED_FROM_SHELF = "PICKED_FROM_SHELF"
    IN_HAND = "IN_HAND"
    BEING_INSPECTED = "BEING_INSPECTED"
    TRACKING_UNCERTAIN = "TRACKING_UNCERTAIN"
    RETURN_CANDIDATE = "RETURN_CANDIDATE"
    RETURNED_TO_SHELF = "RETURNED_TO_SHELF"
    PLACED_IN_BAG = "PLACED_IN_BAG"
    MISSING_SUSPICIOUS = "MISSING_SUSPICIOUS"
    THEFT_ALERT = "THEFT_ALERT"
    CUSTOMER_EXITED_UNRESOLVED = "CUSTOMER_EXITED_UNRESOLVED"


# ────────────────────────── DETECTOR (YOLOv8 & POSE) ─────────────────────────
def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


class DetectorWrapper:
    def __init__(self, model_det="yolov8n.pt", model_pose="yolov8n-pose.pt", imgsz=480):
        self.device = get_device()
        self.imgsz = imgsz
        print(f"[Detector] Initializing using device: {self.device.upper()}")
        
        det_path = model_det if os.path.isabs(model_det) else os.path.join(HERE, model_det)
        pose_path = model_pose if os.path.isabs(model_pose) else os.path.join(HERE, model_pose)

        from ultralytics import YOLO
        print(f"[Detector] Loading models: {os.path.basename(det_path)}, {os.path.basename(pose_path)}")
        self.model_det = YOLO(det_path)
        self.model_pose = YOLO(pose_path)

    def detect_objects_and_pose(self, frame, conf_det=0.30, conf_pose=0.25):
        persons = []
        containers = []

        res_det = self.model_det.predict(
            frame, conf=conf_det, device=self.device, imgsz=self.imgsz,
            verbose=False, classes=[0, 24, 26, 28]  # person, backpack, handbag, suitcase
        )
        for r in res_det:
            boxes = r.boxes
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls[0].item())
                cf = float(b.conf[0].item())
                xyxy = b.xyxy[0].cpu().numpy().tolist()
                lbl = r.names.get(cls_id, str(cls_id)).upper()

                if cls_id == 0:
                    persons.append({"box": xyxy, "conf": cf, "k": np.zeros((17, 3), dtype=np.float32)})
                else:
                    containers.append({"box": xyxy, "conf": cf, "label": lbl})

        if persons:
            res_pose = self.model_pose.predict(
                frame, conf=conf_pose, device=self.device, imgsz=self.imgsz,
                verbose=False, classes=[0]
            )
            for r in res_pose:
                if r.keypoints is None or r.boxes is None:
                    continue
                kpts_data = r.keypoints.data.cpu().numpy()
                pose_boxes = r.boxes.xyxy.cpu().numpy()
                for pb, pk in zip(pose_boxes, kpts_data):
                    best_iou, best_idx = 0.25, None
                    for pi, p_obj in enumerate(persons):
                        cur_iou = iou(p_obj["box"], pb)
                        if cur_iou > best_iou:
                            best_iou = cur_iou
                            best_idx = pi
                    if best_idx is not None:
                        persons[best_idx]["k"] = pk

        return persons, containers


# ────────────────────────── UTILITY FUNCTIONS ────────────────────────────────
def iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def box_center(b):
    return (float((b[0] + b[2]) / 2.0), float((b[1] + b[3]) / 2.0))


def point_in_box(pt, box, margin=0):
    return (box[0] - margin <= pt[0] <= box[2] + margin) and (box[1] - margin <= pt[1] <= box[3] + margin)


def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def detect_shelf_gaps(frame, clean_bg, shelf_mask, threshold=30, min_area=450, person_mask=None):
    """
    Identifies actual merchandise removed from the shelf ROI by comparing the current frame
    against the clean shelf background reference, masking out human occlusion.
    """
    if clean_bg is None or frame is None:
        return []

    diff = cv2.absdiff(frame, clean_bg)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    masked_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=shelf_mask)

    if person_mask is not None:
        masked_diff = cv2.bitwise_and(masked_diff, cv2.bitwise_not(person_mask))

    _, thresh = cv2.threshold(masked_diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gaps = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= min_area:
            bx, by, bw, bh = cv2.boundingRect(c)
            gaps.append({
                "box": [bx, by, bx + bw, by + bh],
                "area": area,
                "center": (bx + bw / 2.0, by + bh / 2.0)
            })

    gaps.sort(key=lambda x: -x["area"])
    return gaps


# ────────────────────── PHYSICAL PRODUCT TRACKER ────────────────────────────
class PhysicalProductTracker:
    """
    Tracks the actual physical retail product independent of the customer's hands.
    - Locks appearance template and dimensions from clean shelf background at pickup.
    - Tracks merchandise using multi-scale normalized cross-correlation.
    - Never assigns wrist coordinates to the product.
    - Accurately reports occlusions without synthesizing fake boxes.
    """
    def __init__(self, prod_id, customer_id, shelf_box, template, t_pick):
        self.id = prod_id
        self.customer_id = customer_id
        self.shelf_box = list(shelf_box)
        self.current_box = list(shelf_box)   # [x1, y1, x2, y2]
        self.template = template.copy() if template is not None else None
        self.t_pick = t_pick
        self.t_last_seen = t_pick
        self.state = ProductState.PICKED_FROM_SHELF
        self.confidence = 1.0
        self.is_visually_verified = True
        self.occluded_frames = 0
        self.return_candidate_frames = 0
        self.bag_drop_candidate_frames = 0
        self.orig_w = max(24, shelf_box[2] - shelf_box[0])
        self.orig_h = max(35, shelf_box[3] - shelf_box[1])

    def update_visual_tracking(self, frame, customer_box, wrist_pts):
        """
        Visually searches for and tracks the physical merchandise across the customer's
        hands and upper body region. Separates product from hand.
        """
        if self.template is None or self.template.size == 0:
            return False

        H, W = frame.shape[:2]
        cb = customer_box
        ph = max(20.0, cb[3] - cb[1])

        # Center search near wrists if visible, else customer chest/torso
        if wrist_pts:
            cx = float(np.mean([p[0] for p in wrist_pts]))
            cy = float(np.mean([p[1] for p in wrist_pts]))
        else:
            cx = (cb[0] + cb[2]) / 2.0
            cy = cb[1] + 0.35 * ph

        search_rad = int(0.18 * ph)
        sx1 = max(0, int(cx - search_rad))
        sy1 = max(0, int(cy - search_rad))
        sx2 = min(W, int(cx + search_rad))
        sy2 = min(H, int(cy + search_rad))

        if (sx2 - sx1) < (self.orig_w + 10) or (sy2 - sy1) < (self.orig_h + 10):
            self.is_visually_verified = False
            return False

        search_patch = frame[sy1:sy2, sx1:sx2]
        sw_h, sw_w = search_patch.shape[:2]

        best_val = -1.0
        best_box = None

        # Multi-scale normalized cross-correlation
        for scale in [0.75, 0.85, 1.0, 1.15]:
            tw = int(self.orig_w * scale)
            th = int(self.orig_h * scale)
            if tw >= 16 and th >= 25 and tw < sw_w and th < sw_h:
                t_scaled = cv2.resize(self.template, (tw, th))
                res = cv2.matchTemplate(search_patch, t_scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best_val:
                    best_val = max_val
                    best_box = [
                        sx1 + max_loc[0],
                        sy1 + max_loc[1],
                        sx1 + max_loc[0] + tw,
                        sy1 + max_loc[1] + th
                    ]

        # Verification threshold: Product is visually verified
        if best_val >= 0.32 and best_box is not None:
            # Smooth bounding box
            self.current_box = [
                int(0.65 * self.current_box[0] + 0.35 * best_box[0]),
                int(0.65 * self.current_box[1] + 0.35 * best_box[1]),
                int(0.65 * self.current_box[2] + 0.35 * best_box[2]),
                int(0.65 * self.current_box[3] + 0.35 * best_box[3]),
            ]
            self.confidence = float(best_val)
            self.is_visually_verified = True
            self.occluded_frames = 0
            return True
        else:
            # Product is occluded by fingers/body or customer is turned
            self.confidence = float(max(0.0, best_val))
            self.is_visually_verified = False
            self.occluded_frames += 1
            return False


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
        self.active_product = None        # PhysicalProductTracker instance
        self.completed_products = []
        self.theft_flagged = False
        self.bag_box = None
        self.bag_type = "BAG"


# ────────────────────────── INTERACTIVE SHELF DRAWER ────────────────────────
def draw_shelf_roi(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ret, frame_native = cap.read()
    cap.release()

    if not ret or frame_native is None:
        raise RuntimeError(f"Cannot read frame from {video_path}")

    H, W = frame_native.shape[:2]
    dw, dh = 1280, 720
    sx, sy = dw / W, dh / H
    disp = cv2.resize(frame_native, (dw, dh))

    polys = []
    cur = []
    mouse = [-1, -1]

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

    win = "Draw Shelf ROI  |  L-Click: Add Point  |  R-Click: Close  |  D: Default  |  Enter: Run"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    cv2.setMouseCallback(win, mouse_callback)

    while True:
        vis = disp.copy()
        for poly in polys:
            pts = np.array(poly, np.int32)
            cv2.polylines(vis, [pts], True, (0, 220, 255), 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], (0, 180, 255))
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

        if len(cur) > 0:
            for pt in cur:
                cv2.circle(vis, pt, 4, (0, 255, 0), -1)
            for i in range(len(cur) - 1):
                cv2.line(vis, cur[i], cur[i+1], (0, 255, 0), 2)
            if mouse[0] >= 0:
                cv2.line(vis, cur[-1], tuple(mouse), (0, 255, 255), 1)

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
        elif key in (13, ord('s'), ord('S'), 32):
            if len(cur) >= 3:
                polys.append(list(cur))
                cur.clear()
            if polys:
                break
            elif default_shelves:
                polys = [list(poly) for poly in default_shelves]
                break
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

    shelf_np = [np.array(p, np.int32) for p in shelf_polygons]
    shelf_mask = np.zeros((H, W), dtype=np.uint8)
    for sp in shelf_np:
        cv2.fillPoly(shelf_mask, [sp], 255)

    # Initialize clean reference shelf background
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ret, clean_shelf_bg = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, clean_shelf_bg = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("[Pipeline] Clean shelf ROI background initialized successfully.")

    model_det = f"yolov8{model_size}.pt"
    model_pose = f"yolov8{model_size}-pose.pt"
    detector = DetectorWrapper(model_det=model_det, model_pose=model_pose, imgsz=480)

    out_file = os.path.join(HERE, "out_theft_v2.mp4")
    wr = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (total_w, 720)) if save else None

    if show:
        cv2.namedWindow("AI Retail Theft Detection v2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Retail Theft Detection v2", total_w, 720)

    people = {}
    next_customer_id = 1
    events = []
    accounted_shelf_gaps = []

    # Accurate 7-Metric Dashboard
    counts = dict(
        customers_in_view=0,
        total_picked=0,
        currently_held=0,
        returned_to_shelf=0,
        placed_in_bag=0,
        suspicious_missing=0,
        theft_alerts=0
    )

    active_theft_banner_until_t = 0.0

    def log_event(t, text, col):
        events.append((t, text, col))
        prefix = "🚨 " if col is RED else ("⚠️ " if col is ORANGE else ("🛍 " if col is AMB else "✓ "))
        print(f"[{t:5.1f}s] {prefix}{text}")

    idx = 0
    t_prev = time.time()
    fps_disp = 0.0
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

        # 1. RUN YOLO INFERENCE
        if idx % (skip_frames + 1) == 1 or not cached_persons:
            persons_det, containers_det = detector.detect_objects_and_pose(frame, conf_det=0.30, conf_pose=0.25)
            cached_persons = persons_det
            cached_containers = containers_det
        else:
            persons_det = cached_persons
            containers_det = cached_containers

        # 2. SHELF GAP ANALYSIS (GLOBAL FOR FRAME)
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

        # 3. PERSISTENT MULTI-PERSON TRACKING
        matched_det_indices = set()
        active_customer_tracks = []

        for cid, trk in sorted(people.items()):
            if trk.lost > 60:
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
                if trk.lost <= 15:
                    active_customer_tracks.append(trk)

        # Register genuine new customers
        for j, d in enumerate(persons_det):
            if j not in matched_det_indices:
                db = d["box"]
                dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
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

        counts["customers_in_view"] = len(active_customer_tracks)

        # 4. PROCESS INTERACTIONS FOR EACH ACTIVE CUSTOMER
        for p in active_customer_tracks:
            b = p.box
            k = p.kpts
            ph = max(1.0, b[3] - b[1])
            reach_margin = int(0.06 * ph)

            ls, rs = k[5], k[6]
            lh, rh = k[11], k[12]
            sh_valid = [q for q in (ls, rs) if q[2] > 0.15]
            hip_valid = [q for q in (lh, rh) if q[2] > 0.15]
            sy_mean = float(np.mean([q[1] for q in sh_valid])) if sh_valid else b[1] + 0.18 * ph
            hy_mean = float(np.mean([q[1] for q in hip_valid])) if hip_valid else sy_mean + 0.42 * ph

            # Active Bag Detection
            p.bag_box = None
            for c in containers_det:
                cb = c["box"]
                ccx, ccy = box_center(cb)
                if not any(point_in_poly((ccx, ccy), s) for s in shelf_np):
                    if (b[0] - 0.20 * ph) <= ccx <= (b[2] + 0.20 * ph) and sy_mean - 0.15 * ph <= ccy <= hy_mean + 0.25 * ph:
                        p.bag_box = [int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3])]
                        p.bag_type = c.get("label", "BAG")
                        break

            # Explicit Hand Detection (Wrists)
            hand_in_shelf = False
            wrist_pts_valid = []
            for w_idx in (9, 10):
                wkp = k[w_idx]
                if wkp[2] > 0.18:
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

            # 5. PRODUCT LIFECYCLE & PHYSICAL TRACKING
            if p.active_product is None:
                # ── STAGE A: PHYSICAL PRODUCT PICKUP CONFIRMATION ──
                # Confirmed when: Hand was at shelf AND pulled away AND a genuine gap formed!
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
                                prod_id = counts["total_picked"] + 1
                                accounted_shelf_gaps.append((gcx, gcy))
                                gx1, gy1, gx2, gy2 = g["box"]
                                
                                # Extract physical merchandise template from reference background
                                prod_crop = clean_shelf_bg[gy1:gy2, gx1:gx2].copy()
                                item_w = gx2 - gx1
                                item_h = gy2 - gy1

                                # Initialize dedicated Physical Product Tracker
                                p.active_product = PhysicalProductTracker(
                                    prod_id=prod_id,
                                    customer_id=p.id,
                                    shelf_box=g["box"],
                                    template=prod_crop,
                                    t_pick=t
                                )
                                p.active_product.state = ProductState.PICKED_FROM_SHELF
                                p.state = "HOLDING PRODUCT"
                                counts["total_picked"] += 1
                                counts["currently_held"] += 1
                                log_event(t, f"Customer #{p.id}: Product #{prod_id} PICKED from shelf [Item: {item_w}x{item_h}px]", AMB)
                                break

                if hand_in_shelf:
                    p.state = "REACHING SHELF"
                elif not p.theft_flagged:
                    p.state = "BROWSING"

            else:
                # ── STAGE B: PHYSICAL MERCHANDISE VISUAL TRACKING ──
                prod = p.active_product
                
                # Update visual tracking of actual physical item (NOT hand)
                visually_verified = prod.update_visual_tracking(frame, p.box, wrist_pts_valid)
                
                # Check customer posture: inspecting in front of chest
                wrists_at_chest = any(
                    (b[0] - 0.15 * ph) <= w[0] <= (b[2] + 0.15 * ph) and
                    (sy_mean - 0.10 * ph) <= w[1] <= (hy_mean + 0.25 * ph)
                    for w in wrist_pts_valid
                )

                if visually_verified:
                    prod.t_last_seen = t
                    prod.state = ProductState.IN_HAND
                    p.state = "HOLDING PRODUCT"
                elif wrists_at_chest:
                    # Physically held in grasp, visually occluded by fingers/palms
                    prod.t_last_seen = t
                    prod.state = ProductState.BEING_INSPECTED
                    p.state = "HOLDING PRODUCT"
                else:
                    prod.state = ProductState.TRACKING_UNCERTAIN
                    p.state = "PRODUCT OCCLUDED / UNCERTAIN"

                # ── STAGE C: PRODUCT RESOLUTIONS (RETURN, BAG, THEFT) ──
                # Condition 1: RETURN TO SHELF
                hand_back_on_shelf = hand_in_shelf or any(
                    cv2.pointPolygonTest(s, w_pt, True) >= -reach_margin
                    for s in shelf_np
                    for w_pt in wrist_pts_valid
                )

                if hand_back_on_shelf:
                    prod.state = ProductState.RETURN_CANDIDATE
                    prod.return_candidate_frames += 1
                    if prod.return_candidate_frames >= 4:
                        prod.state = ProductState.RETURNED_TO_SHELF
                        p.state = "PUT BACK ON SHELF"
                        counts["returned_to_shelf"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        if counts["suspicious_missing"] > 0:
                            counts["suspicious_missing"] -= 1
                        log_event(t, f"Customer #{p.id}: Product #{prod.id} RETURNED TO SHELF [Put Back: {counts['returned_to_shelf']}]", GRN)
                        
                        # Clear gap from accounted list
                        orig_c = box_center(prod.shelf_box)
                        accounted_shelf_gaps = [ag for ag in accounted_shelf_gaps if np.hypot(ag[0]-orig_c[0], ag[1]-orig_c[1]) >= 70]
                        
                        p.completed_products.append(prod)
                        p.active_product = None

                # Condition 2: BAG PLACEMENT (SAFE)
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

                # Condition 3: EVIDENCE-BASED THEFT
                else:
                    time_unseen = t - prod.t_last_seen
                    if time_unseen > 4.0:
                        if prod.state != ProductState.MISSING_SUSPICIOUS:
                            prod.state = ProductState.MISSING_SUSPICIOUS
                            p.state = "PRODUCT MISSING (SUSPICIOUS)"
                            counts["suspicious_missing"] += 1
                            log_event(t, f"Customer #{p.id}: Product #{prod.id} TRACKING UNCERTAIN / MISSING", ORANGE)

                    if time_unseen > 15.0 and not hand_in_shelf:
                        prod.state = ProductState.THEFT_ALERT
                        p.theft_flagged = True
                        p.state = "THEFT ALERT (ITEM MISSING)"
                        counts["theft_alerts"] += 1
                        counts["currently_held"] = max(0, counts["currently_held"] - 1)
                        if counts["suspicious_missing"] > 0:
                            counts["suspicious_missing"] -= 1
                        active_theft_banner_until_t = t + 6.0
                        log_event(t, f"Customer #{p.id}: 🚨 THEFT ALERT! Product #{prod.id} MISSING (NOT RETURNED TO SHELF)", RED)
                        p.completed_products.append(prod)
                        p.active_product = None

        # Exited customer check
        for cid, p in list(people.items()):
            if t - p.t > 2.5 and p.lost > 60:
                if p.active_product is not None:
                    prod = p.active_product
                    prod.state = ProductState.CUSTOMER_EXITED_UNRESOLVED
                    counts["currently_held"] = max(0, counts["currently_held"] - 1)
                    counts["theft_alerts"] += 1
                    active_theft_banner_until_t = t + 6.0
                    log_event(t, f"Customer #{p.id} EXITED AISLE WITH UNPAID PRODUCT #{prod.id}!", RED)
                    p.completed_products.append(prod)
                    p.active_product = None

        # 6. RENDER VIDEO WITH STRICT HAND & PRODUCT SEPARATION
        render_frame = cv2.resize(frame, (vw, 720))

        # Draw shelf overlay
        for sp in shelf_polygons:
            poly_scaled = np.array([[(int(pt[0] * sx_render), int(pt[1] * sy_render)) for pt in sp]], np.int32)
            cv2.polylines(render_frame, poly_scaled, True, SHELF_COLOR, 2)
            cv2.putText(render_frame, "ACTIVE SHELF ROI [SHELF #1]", (poly_scaled[0][0][0] + 8, poly_scaled[0][0][1] + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, SHELF_COLOR, 2)

        # Draw customers, hands, and products
        for p in active_customer_tracks:
            b_sc = [int(p.box[0] * sx_render), int(p.box[1] * sy_render),
                    int(p.box[2] * sx_render), int(p.box[3] * sy_render)]
            box_col = RED if p.theft_flagged else (AMB if p.active_product is not None else GRN)
            cv2.rectangle(render_frame, (b_sc[0], b_sc[1]), (b_sc[2], b_sc[3]), box_col, 2)

            cust_label = f"CUSTOMER #{p.id} | {p.state}"
            cv2.putText(render_frame, cust_label, (b_sc[0], max(20, b_sc[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_col, 2)

            # Draw Customer HAND Bounding Boxes (Distinct from Product)
            for widx, wname in ((9, "L"), (10, "R")):
                wkp = p.kpts[widx]
                if wkp[2] > 0.18:
                    wx_sc, wy_sc = int(wkp[0] * sx_render), int(wkp[1] * sy_render)
                    # Hand bounding box
                    cv2.rectangle(render_frame, (wx_sc - 18, wy_sc - 18), (wx_sc + 18, wy_sc + 18), HAND_COLOR, 2)
                    cv2.putText(render_frame, f"HAND [{wname}] | CUST #{p.id}", (wx_sc - 35, max(14, wy_sc - 22)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, HAND_COLOR, 1)

            # Draw ACTUAL PHYSICAL PRODUCT Bounding Box (Distinct from Hands)
            if p.active_product is not None:
                ap = p.active_product
                if ap.is_visually_verified:
                    pb = ap.current_box
                    pb_sc = [int(pb[0] * sx_render), int(pb[1] * sy_render),
                             int(pb[2] * sx_render), int(pb[3] * sy_render)]
                    
                    prod_col = RED if ap.state == ProductState.THEFT_ALERT else (
                        ORANGE if ap.state in (ProductState.TRACKING_UNCERTAIN, ProductState.MISSING_SUSPICIOUS) else PROD_COLOR
                    )
                    cv2.rectangle(render_frame, (pb_sc[0], pb_sc[1]), (pb_sc[2], pb_sc[3]), prod_col, 2)
                    prod_tag = f"PRODUCT #{ap.id} | CUST #{p.id} | CONF: {ap.confidence:.2f} [{ap.state}]"
                    cv2.putText(render_frame, prod_tag, (pb_sc[0] - 20, max(16, pb_sc[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, prod_col, 2)
                else:
                    # When occluded, display status tag without drawing a fake box
                    cv2.putText(render_frame, f"[PRODUCT #{ap.id}: OCCLUDED / UNCERTAIN]",
                                (b_sc[0], b_sc[3] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, ORANGE, 2)

        # 7. DASHBOARD TELEMETRY PANEL
        panel = np.zeros((720, PW, 3), dtype=np.uint8)
        panel[:] = PANEL_BG

        cv2.putText(panel, "AI RETAIL SURVEILLANCE", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TXT_WHITE, 2)
        cv2.line(panel, (20, 50), (PW - 20, 50), BORDER_COLOR, 1)

        y_off = 78
        metrics = [
            ("CUSTOMERS IN VIEW", counts["customers_in_view"], TXT_WHITE),
            ("TOTAL PRODUCTS PICKED", counts["total_picked"], AMB),
            ("CURRENTLY IN HAND", counts["currently_held"], (255, 200, 50)),
            ("RETURNED TO SHELF", counts["returned_to_shelf"], GRN),
            ("PLACED IN BAG", counts["placed_in_bag"], GRN),
            ("SUSPICIOUS MISSING", counts["suspicious_missing"], ORANGE),
            ("THEFT ALERTS", counts["theft_alerts"], RED if counts["theft_alerts"] > 0 else TXT_MUTED)
        ]
        for title, val, col in metrics:
            cv2.putText(panel, title, (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.40, TXT_MUTED, 1)
            cv2.putText(panel, str(val), (PW - 70, y_off + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.68, col, 2)
            y_off += 36

        # Recent Activity Log
        y_off += 10
        cv2.line(panel, (20, y_off), (PW - 20, y_off), BORDER_COLOR, 1)
        y_off += 25
        cv2.putText(panel, "ACTIVITY LOG", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TXT_WHITE, 1)
        y_off += 20

        for ev_t, ev_txt, ev_col in events[-8:]:
            txt_disp = f"[{ev_t:4.1f}s] {ev_txt[:34]}"
            cv2.putText(panel, txt_disp, (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ev_col, 1)
            y_off += 22

        # Theft banner overlay
        is_banner_active = (t < active_theft_banner_until_t)
        if is_banner_active:
            cv2.rectangle(render_frame, (0, 0), (vw, 55), RED, -1)
            cv2.putText(render_frame, "🚨 THEFT DETECTED: MERCHANDISE CONCEALED", (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        combined = np.hstack([render_frame, panel])

        if wr is not None:
            wr.write(combined)

        if show:
            cv2.imshow("AI Retail Theft Detection v2", combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            elif key == 32:
                cv2.waitKey(0)

    cap.release()
    if wr is not None:
        wr.release()
        print(f"\n[Pipeline] Processed output video saved to: {out_file}")
    if show:
        cv2.destroyAllWindows()

    print("\n" + "="*25 + " FINAL REPORT " + "="*25)
    for k, v in counts.items():
        print(f"  {k:<25} : {v}")
    print("="*64 + "\n")


# ─────────────────────────── CLI ENTRY POINT ────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AI Retail Theft Detection v2 (vid2.mp4 & Retail Surveillance)")
    parser.add_argument("video", nargs="?", default="vid2.mp4", help="Video path")
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
