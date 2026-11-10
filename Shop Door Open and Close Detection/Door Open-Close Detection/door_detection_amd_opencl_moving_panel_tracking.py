import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime
import pyopencl as cl

# AMD Radeon R5 M430 / Hainan OpenCL door detector
VIDEO_PATH = "Footage3.mp4"
LOG_FILE = "door_events_opencl.csv"
STABLE_FRAMES = 8
MAX_MATCH_DISTANCE = 35.0
MIN_MATCH_GAP = 2.0
FILL_ALPHA = 0.35
FRAME_SKIP = 1                 # Keep 1 initially for reliable detection
DISPLAY_WAIT_MS = 1            # Faster display refresh
DISPLAY_SCALE = 1.0
LOCAL_SIZE = 256

# Door-panel tracking (CPU optical flow; comparison remains on AMD OpenCL)
TRACK_MAX_CORNERS = 120
TRACK_MIN_POINTS = 6
TRACK_MIN_INLIERS = 5
TRACK_QUALITY = 0.01
TRACK_MIN_DISTANCE = 5
TRACK_MAX_TRANSLATION = 0.25  # fraction of frame size per frame

KERNEL = r"""
__kernel void masked_absdiff_reduce(
    __global const uchar *current,
    __global const uchar *reference,
    __global const uchar *mask,
    __global float *partial_sums,
    const uint n,
    __local float *scratch)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group = get_group_id(0);
    float value = 0.0f;

    if (gid < n && mask[gid] != 0)
        value = fabs((float)current[gid] - (float)reference[gid]);

    scratch[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            scratch[lid] += scratch[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        partial_sums[group] = scratch[0];
}
"""

try:
    import winsound
    def play_alert(state):
        winsound.Beep(1500 if state == "OPEN" else 900, 450)
except ImportError:
    def play_alert(state):
        print("\a", end="", flush=True)


def make_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return np.ascontiguousarray(gray)


def init_amd():
    for platform in cl.get_platforms():
        if "AMD" in platform.name.upper():
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if devices:
                device = devices[0]
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx)
                program = cl.Program(ctx, KERNEL).build()
                print("AMD OpenCL platform:", platform.name)
                print("Selected GPU:", device.name)
                print("Vendor:", device.vendor)
                print("OpenCL version:", device.version)
                return ctx, queue, program
    raise RuntimeError("AMD OpenCL GPU not found.")




def order_quad_points(points):
    """Return four polygon vertices in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError("Each sliding door must be marked with exactly 4 corner points.")

    sums = pts.sum(axis=1)
    # x-y is used here: the top-right corner has the largest value,
    # while the bottom-left corner has the smallest value.
    diffs = pts[:, 0] - pts[:, 1]

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmax(diffs)]
    bottom_left = pts[np.argmin(diffs)]

    ordered = np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )

    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("Could not order the four door corners. Mark them again.")

    return ordered


class DoorPanelTracker:
    """
    Track a sliding panel using two independent feature clusters on its
    LEFT vertical frame edge (top-left and bottom-left polygon corners).

    This avoids re-detecting features across the entire frosted-glass panel,
    where subtitles, reflections and the adjacent panel can cause drift.
    The panel is assumed to translate horizontally in a fixed-camera video.
    """

    CORNER_RADIUS = 8
    GRID_STEP = 4
    MIN_POINTS_PER_CORNER = 4
    MAX_STEP_X = 12.0
    MAX_STEP_Y = 5.0
    MAX_CORNER_DISAGREEMENT = 4.5

    def __init__(self, first_frame, base_points):
        self.height, self.width = first_frame.shape[:2]
        self.base_points = order_quad_points(base_points).astype(np.float32)
        self.matrix = np.eye(3, dtype=np.float64)
        self.previous_gray = make_gray(first_frame)
        self.corner_points = self._make_corner_points()
        self.tracking_ok = all(
            len(p) >= self.MIN_POINTS_PER_CORNER for p in self.corner_points
        )

    def _make_mask(self, points):
        """Build a safe polygon mask for the currently tracked panel."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        try:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                return mask
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 3:
                return mask
            pts[:, 0] = np.clip(pts[:, 0], 0, self.width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, self.height - 1)
            polygon = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
            if abs(cv2.contourArea(polygon)) < 4:
                return mask
            cv2.fillPoly(mask, [polygon], 255)
        except (ValueError, TypeError, cv2.error, OverflowError):
            return np.zeros((self.height, self.width), dtype=np.uint8)
        return mask

    def _make_corner_points(self):
        # Polygon order: TL, TR, BR, BL.
        # Track only the panel's left edge corners (TL and BL).
        clusters = []
        for corner_index in (0, 3):
            cx, cy = self.base_points[corner_index]
            pts = []
            for dx in range(-self.CORNER_RADIUS,
                            self.CORNER_RADIUS + 1, self.GRID_STEP):
                for dy in range(-self.CORNER_RADIUS,
                                self.CORNER_RADIUS + 1, self.GRID_STEP):
                    x = np.clip(cx + dx, 2, self.width - 3)
                    y = np.clip(cy + dy, 2, self.height - 3)
                    pts.append([x, y])
            clusters.append(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2))
        return clusters

    @staticmethod
    def _robust_cluster_delta(old_points, new_points):
        if len(old_points) < 3 or len(new_points) < 3:
            return None, None

        deltas = new_points.reshape(-1, 2) - old_points.reshape(-1, 2)
        median = np.median(deltas, axis=0)
        distances = np.linalg.norm(deltas - median, axis=1)
        inliers = distances <= 3.0

        if int(inliers.sum()) < DoorPanelTracker.MIN_POINTS_PER_CORNER:
            return None, None

        delta = np.median(deltas[inliers], axis=0)
        retained = new_points.reshape(-1, 2)[inliers]
        return delta.astype(np.float32), retained.reshape(-1, 1, 2)

    def update(self, frame):
        gray = make_gray(frame)
        new_clusters = []
        cluster_deltas = []

        for points in self.corner_points:
            if len(points) < 3:
                new_clusters.append(points)
                cluster_deltas.append(None)
                continue

            next_points, status, errors = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                points,
                None,
                winSize=(31, 31),
                maxLevel=4,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01
                )
            )

            if next_points is None or status is None:
                new_clusters.append(points)
                cluster_deltas.append(None)
                continue

            good = status.reshape(-1) == 1
            old_good = points[good]
            new_good = next_points[good]

            delta, retained = self._robust_cluster_delta(
                old_good, new_good
            )

            if delta is None:
                new_clusters.append(old_good if len(old_good) else points)
                cluster_deltas.append(None)
            else:
                new_clusters.append(retained)
                cluster_deltas.append(delta)

        self.previous_gray = gray
        self.corner_points = new_clusters

        valid_deltas = [d for d in cluster_deltas if d is not None]

        # Both top and bottom points on the same panel edge must agree.
        if len(valid_deltas) < 2:
            self.tracking_ok = False
            return False

        disagreement = float(np.linalg.norm(
            valid_deltas[0] - valid_deltas[1]
        ))

        step = np.median(np.vstack(valid_deltas), axis=0)
        dx, dy = float(step[0]), float(step[1])

        if (
            disagreement > self.MAX_CORNER_DISAGREEMENT
            or abs(dx) > self.MAX_STEP_X
            or abs(dy) > self.MAX_STEP_Y
        ):
            self.tracking_ok = False
            return False

        # Sliding doors move horizontally. Keep only a very small vertical
        # correction to suppress optical-flow jitter.
        dy = float(np.clip(dy, -1.5, 1.5))

        self.matrix[0, 2] += dx
        self.matrix[1, 2] += dy
        self.tracking_ok = True
        return True

    def current_points(self):
        points = self.base_points.copy()
        points[:, 0] += float(self.matrix[0, 2])
        points[:, 1] += float(self.matrix[1, 2])
        points[:, 0] = np.clip(points[:, 0], 0, self.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.height - 1)
        return points.astype(np.float32)

    def reset(self, first_frame):
        self.matrix = np.eye(3, dtype=np.float64)
        self.previous_gray = make_gray(first_frame)
        self.corner_points = self._make_corner_points()
        self.tracking_ok = all(
            len(p) >= self.MIN_POINTS_PER_CORNER for p in self.corner_points
        )


class AMDComparator:
    def __init__(self, ctx, queue, program, shape):
        self.ctx, self.queue, self.program = ctx, queue, program
        self.height, self.width = shape
        self.n = self.height * self.width
        self.groups = (self.n + LOCAL_SIZE - 1) // LOCAL_SIZE
        self.global_size = self.groups * LOCAL_SIZE
        mf = cl.mem_flags
        self.current = cl.Buffer(ctx, mf.READ_ONLY, self.n)
        self.partial = cl.Buffer(ctx, mf.WRITE_ONLY,
                                 self.groups * np.dtype(np.float32).itemsize)
        self.partial_host = np.empty(self.groups, np.float32)
        self.gpu_doors = {}

    def add_door(self, door):
        mf = cl.mem_flags
        mask = np.ascontiguousarray(door["mask"], dtype=np.uint8)
        open_ref = make_gray(door["open_ref"])
        closed_ref = make_gray(door["closed_ref"])
        count = int(np.count_nonzero(mask))
        if count == 0:
            raise ValueError(f'{door["id"]} has an empty mask.')

        self.gpu_doors[door["id"]] = {
            "mask": cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                              hostbuf=mask),
            "open_ref": cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                  hostbuf=open_ref),
            "closed_ref": cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                                    hostbuf=closed_ref),
            "count": count
        }

    def upload_frame(self, frame):
        gray = make_gray(frame)
        cl.enqueue_copy(self.queue, self.current, gray)
        return gray


    def update_door_geometry(self, door):
        """Upload the tracked polygon mask and pose-aligned references to AMD."""
        item = self.gpu_doors[door["id"]]
        tracker = door["tracker"]
        h, w = self.height, self.width

        points = tracker.current_points()
        mask = tracker._make_mask(points)
        count = int(np.count_nonzero(mask))

        if count == 0 or len(points) < 3:
            door["tracking_ok"] = False
            return

        polygon = np.rint(points).astype(np.int32).reshape((-1, 1, 2))

        # Transform each calibration image from its saved panel pose to
        # the panel's current tracked pose.
        try:
            current_inv_ref_open = tracker.matrix @ np.linalg.inv(
                door["open_ref_transform"]
            )
            current_inv_ref_closed = tracker.matrix @ np.linalg.inv(
                door["closed_ref_transform"]
            )
        except np.linalg.LinAlgError:
            door["tracking_ok"] = False
            return

        open_aligned = cv2.warpAffine(
            door["open_ref_gray"],
            current_inv_ref_open[:2, :],
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        closed_aligned = cv2.warpAffine(
            door["closed_ref_gray"],
            current_inv_ref_closed[:2, :],
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        mask = np.ascontiguousarray(mask)
        open_aligned = np.ascontiguousarray(open_aligned)
        closed_aligned = np.ascontiguousarray(closed_aligned)

        cl.enqueue_copy(self.queue, item["mask"], mask)
        cl.enqueue_copy(self.queue, item["open_ref"], open_aligned)
        cl.enqueue_copy(self.queue, item["closed_ref"], closed_aligned)
        self.queue.finish()

        item["count"] = count
        door["points"] = points.tolist()
        door["mask"] = mask
        door["bbox"] = cv2.boundingRect(polygon)
        door["tracking_ok"] = tracker.tracking_ok


    def difference(self, door_id, reference_key):
        item = self.gpu_doors[door_id]
        self.program.masked_absdiff_reduce(
            self.queue, (self.global_size,), (LOCAL_SIZE,),
            self.current, item[reference_key], item["mask"],
            self.partial, np.uint32(self.n),
            cl.LocalMemory(LOCAL_SIZE * np.dtype(np.float32).itemsize)
        )
        cl.enqueue_copy(self.queue, self.partial_host, self.partial).wait()
        return float(np.sum(self.partial_host)) / item["count"]


def save_event(door_id, old_state, new_state, open_score, closed_score):
    now = datetime.now()
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Date", "Time", "Door ID", "Previous State",
                             "New State", "Open Match Score",
                             "Closed Match Score"])
        writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                         door_id, old_state, new_state,
                         f"{open_score:.2f}", f"{closed_score:.2f}"])
    print(f"{now:%H:%M:%S} | {door_id}: {old_state} -> {new_state}")


def draw_status(frame, door):
    state = door["state"]
    if not door.get("tracking_ok", True):
        color = (0, 165, 255)  # amber: panel tracking uncertain
    else:
        color = ((0, 200, 0) if state == "CLOSED" else
                 (0, 0, 255) if state == "OPEN" else (0, 165, 255))
    pts = np.asarray(door["points"], dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) >= 3:
        pts[:, 0] = np.clip(pts[:, 0], 0, frame.shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, frame.shape[0] - 1)
        polygon = np.rint(pts).astype(np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        frame = cv2.addWeighted(overlay, FILL_ALPHA, frame, 1-FILL_ALPHA, 0)
        cv2.polylines(frame, [polygon], True, color, 3)

    x, y, _, _ = door["bbox"]
    label = f'{door["id"]}  {state}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    label_y = max(th + 12, y - 7)
    cv2.rectangle(frame, (x, label_y-th-9), (x+tw+12, label_y+5),
                  (25, 25, 25), -1)
    cv2.putText(frame, label, (x+6, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    return frame


def select_doors(cap, first_frame):
    h, w = first_frame.shape[:2]
    doors = []
    print("\nMARK EACH DOOR")
    print("Click exactly 4 corners in any order; ENTER saves. The code orders them automatically.")
    print("R resets points; ESC quits. Then type y/n to add another door.")

    while True:
        points = []
        window = f"Mark Door {len(doors)+1}"
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))

        cv2.setMouseCallback(window, mouse_callback)
        while True:
            preview = first_frame.copy()
            for point in points:
                cv2.circle(preview, point, 5, (0, 255, 255), -1)
            if len(points) > 1:
                cv2.polylines(preview, [np.array(points, np.int32)],
                              False, (0, 255, 255), 2)
            cv2.putText(preview, "Mark door | ENTER=save | R=reset",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (0, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                break
            if key == ord("r"):
                points = []
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit("Cancelled.")
        cv2.destroyWindow(window)

        if len(points) != 4:
            print("Please select exactly 4 corner points for one door panel.")
            continue

        try:
            points = order_quad_points(points).tolist()
        except ValueError as exc:
            print(exc)
            continue

        polygon = np.array(points, np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        door = {
            "id": f"DOOR_{len(doors)+1:02d}",
            "points": points,
            "base_points": np.asarray(points, dtype=np.float32),
            "mask": mask,
            "bbox": cv2.boundingRect(polygon),
            "open_ref": None,
            "closed_ref": None,
            "open_ref_gray": None,
            "closed_ref_gray": None,
            "open_ref_transform": None,
            "closed_ref_transform": None,
            "state": "UNKNOWN",
            "candidate": None,
            "stable_count": 0,
            "tracking_ok": True
        }
        door["tracker"] = DoorPanelTracker(first_frame, points)
        doors.append(door)
        print(f'{doors[-1]["id"]} mask saved.')
        if input("Mark another door? (y/n): ").strip().lower() != "y":
            break

    if not doors:
        raise SystemExit("No door masks selected.")
    return doors


def calibrate(cap, first_frame, doors):
    print("\nCALIBRATION: SPACE pause/resume | 1-9 select door")
    print("C save CLOSED | O save OPEN | Q quit")
    print("Each panel is tracked independently from its own left vertical frame edge.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for door in doors:
        door["tracker"].reset(first_frame)

    paused, current, selected = False, first_frame.copy(), 0

    while not all(
        d["open_ref"] is not None and d["closed_ref"] is not None
        for d in doors
    ):
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for door in doors:
                    door["tracker"].reset(first_frame)
                continue

            current = frame
            for door in doors:
                door["tracker"].update(current)
                door["points"] = door["tracker"].current_points().tolist()
                door["tracking_ok"] = door["tracker"].tracking_ok

        preview = current.copy()

        for i, door in enumerate(doors):
            color = (0, 255, 255) if i == selected else (190, 190, 190)
            polygon = np.round(door["points"]).astype(np.int32)
            cv2.polylines(preview, [polygon], True, color, 2)
            # Mark the tracked left-edge anchor corners.
            for anchor in (polygon[0], polygon[3]):
                cv2.circle(preview, tuple(anchor), 5, color, -1)

            bx, by, _, _ = cv2.boundingRect(polygon)
            label = door["id"]
            if not door["tracking_ok"]:
                label += " TRACK?"
            cv2.putText(
                preview, label, (bx, max(22, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

        d = doors[selected]
        closed_text = "SAVED" if d["closed_ref"] is not None else "NEEDED"
        open_text = "SAVED" if d["open_ref"] is not None else "NEEDED"
        tracking_text = "OK" if d["tracking_ok"] else "UNCERTAIN"

        cv2.putText(
            preview,
            f'{d["id"]} | CLOSED: {closed_text} | OPEN: {open_text} | TRACK: {tracking_text}',
            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
        )
        cv2.putText(
            preview,
            "SPACE pause | 1-9 select | C closed | O open | Q quit",
            (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
        )

        cv2.imshow("Door reference calibration", preview)
        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == ord(" "):
            paused = not paused

        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(doors):
                selected = idx

        elif key in (ord("c"), ord("o")):
            if not paused:
                print("Pause the video first, then press C or O.")
                continue
            if not d["tracking_ok"]:
                print("Panel tracking is uncertain. Resume video and let tracking recover before saving.")
                continue

            if key == ord("c"):
                d["closed_ref"] = current.copy()
                d["closed_ref_gray"] = make_gray(current)
                d["closed_ref_transform"] = d["tracker"].matrix.copy()
                cv2.imwrite(f'{d["id"]}_closed.jpg', current)
                print(f'{d["id"]}: CLOSED reference saved at tracked panel position.')
            else:
                d["open_ref"] = current.copy()
                d["open_ref_gray"] = make_gray(current)
                d["open_ref_transform"] = d["tracker"].matrix.copy()
                cv2.imwrite(f'{d["id"]}_open.jpg', current)
                print(f'{d["id"]}: OPEN reference saved at tracked panel position.')

        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Stopped during calibration.")

    cv2.destroyWindow("Door reference calibration")


def main():
    print("TRACKER VERSION: INDEPENDENT HORIZONTAL PANEL TRACKING (NO AFFINE ROTATION)")
    ctx, queue, program = init_amd()

    video_path = VIDEO_PATH
    if not os.path.isabs(video_path):
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open video: {video_path}. Keep the video beside this script "
            "or update VIDEO_PATH."
        )

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Cannot read the first video frame.")

    h, w = first_frame.shape[:2]
    print(f"Video resolution: {w} x {h}")

    doors = select_doors(cap, first_frame)
    calibrate(cap, first_frame, doors)

    # Reset trackers to frame zero so monitoring starts from the same
    # coordinate system used when the reference transforms were recorded.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for door in doors:
        door["tracker"].reset(first_frame)
        door["points"] = door["tracker"].current_points().tolist()
        door["tracking_ok"] = door["tracker"].tracking_ok

    comparator = AMDComparator(ctx, queue, program, (h, w))
    print("\nUploading references and initial masks to AMD GPU...")
    for door in doors:
        comparator.add_door(door)
        print(door["id"], "ready on AMD GPU")

    print(
        "\nAMD GPU monitoring started. "
        "Panel polygons will follow the moving doors. Press Q to stop."
    )

    frame_no = processed = fps_count = 0
    start = fps_start = time.perf_counter()
    shown_fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Video finished.")
            break

        frame_no += 1
        if FRAME_SKIP > 1 and (frame_no - 1) % FRAME_SKIP:
            continue

        # Track each door panel and move its ROI/reference images with it.
        for door in doors:
            door["tracker"].update(frame)
            comparator.update_door_geometry(door)

        comparator.upload_frame(frame)
        display = frame.copy()

        for door in doors:
            if not door.get("tracking_ok", False):
                door["candidate"], door["stable_count"] = None, 0
                door["state"] = "UNKNOWN"
                display = draw_status(display, door)
                continue

            open_score = comparator.difference(door["id"], "open_ref")
            closed_score = comparator.difference(door["id"], "closed_ref")
            best = min(open_score, closed_score)
            gap = abs(open_score - closed_score)

            if best > MAX_MATCH_DISTANCE or gap < MIN_MATCH_GAP:
                detected = "UNKNOWN"
            elif open_score < closed_score:
                detected = "OPEN"
            else:
                detected = "CLOSED"

            if detected == "UNKNOWN":
                door["candidate"], door["stable_count"] = None, 0
            elif detected == door["candidate"]:
                door["stable_count"] += 1
            else:
                door["candidate"], door["stable_count"] = detected, 1

            if (
                door["candidate"] is not None
                and door["stable_count"] >= STABLE_FRAMES
                and door["state"] != door["candidate"]
            ):
                old_state = door["state"]
                door["state"] = door["candidate"]
                save_event(
                    door["id"], old_state, door["state"],
                    open_score, closed_score
                )
                play_alert(door["state"])

            display = draw_status(display, door)

        processed += 1
        fps_count += 1
        now = time.perf_counter()

        if now - fps_start >= 1.0:
            shown_fps = fps_count / (now - fps_start)
            fps_count, fps_start = 0, now

        cv2.putText(
            display, f"AMD OPENCL + PANEL TRACKING | FPS: {shown_fps:.1f}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2
        )
        cv2.putText(
            display,
            "GREEN=CLOSED | RED=OPEN | ORANGE=TRACKING/STATE UNCERTAIN | Q=QUIT",
            (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (255, 255, 255), 2
        )

        shown = display
        if DISPLAY_SCALE != 1.0:
            shown = cv2.resize(
                display, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_AREA
            )

        cv2.imshow("AMD OpenCL - Moving Door Panel Tracking", shown)
        if cv2.waitKey(DISPLAY_WAIT_MS) & 0xFF == ord("q"):
            break

    elapsed = time.perf_counter() - start
    cap.release()
    cv2.destroyAllWindows()

    print("\nStopped. Frames processed:", processed)
    print("Elapsed seconds:", round(elapsed, 2))
    if elapsed > 0:
        print("Average processing FPS:", round(processed / elapsed, 2))
    print("CSV log:", LOG_FILE)


if __name__ == "__main__":
    main()