import cv2
import numpy as np
import csv
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import pyopencl as cl

# AMD Radeon R5 M430 / Hainan OpenCL door detector
VIDEO_PATH = "Footage2.mp4"
LOG_FILE = "door_events_opencl.csv"
CONFIG_DIR = Path("door_configuration")
CONFIG_FILE = CONFIG_DIR / "doors.json"
OPEN_STABLE_FRAMES = 5    # Video frames, NOT wall-clock seconds
CLOSED_STABLE_FRAMES = 8  # Slightly more conservative for closing
SHOW_MATCH_SCORES = True # Shows why a state was selected
MAX_MATCH_DISTANCE = 35.0
MIN_MATCH_GAP = 2.0
FILL_ALPHA = 0.35
FRAME_SKIP = 1                 # Keep 1 initially for reliable detection
DISPLAY_WAIT_MS = 1            # Faster display refresh
DISPLAY_SCALE = 1.0
LOCAL_SIZE = 256

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
    color = ((0, 200, 0) if state == "CLOSED" else
             (0, 0, 255) if state == "OPEN" else (0, 165, 255))
    polygon = np.array(door["points"], dtype=np.int32)
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
    print("Click polygon points around the full door swing area; ENTER saves.")
    print("R resets points; ESC quits. Then type y/n to add another door.")

    while True:
        points = []
        window = f"Mark Door {len(doors)+1}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

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

        if len(points) < 3:
            print("Please select at least 3 points.")
            continue

        polygon = np.array(points, np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        doors.append({
            "id": f"DOOR_{len(doors)+1:02d}", "points": points,
            "mask": mask, "bbox": cv2.boundingRect(polygon),
            "open_ref": None, "closed_ref": None,
            "state": "UNKNOWN", "candidate": None, "stable_count": 0
        })
        print(f'{doors[-1]["id"]} mask saved.')
        if input("Mark another door? (y/n): ").strip().lower() != "y":
            break

    if not doors:
        raise SystemExit("No door masks selected.")
    return doors


def calibrate(cap, first_frame, doors):
    print("\nCALIBRATION: SPACE pause/resume | 1-9 select door")
    print("C save CLOSED | O save OPEN | Q quit")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    paused, current, selected = False, first_frame.copy(), 0

    while not all(d["open_ref"] is not None and d["closed_ref"] is not None
                  for d in doors):
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            current = frame

        preview = current.copy()
        for i, door in enumerate(doors):
            color = (0, 255, 255) if i == selected else (190, 190, 190)
            cv2.polylines(preview, [np.array(door["points"], np.int32)],
                          True, color, 2)
            bx, by, _, _ = door["bbox"]
            cv2.putText(preview, door["id"], (bx, max(22, by-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        d = doors[selected]
        closed_text = "SAVED" if d["closed_ref"] is not None else "NEEDED"
        open_text = "SAVED" if d["open_ref"] is not None else "NEEDED"
        cv2.putText(preview, f'{d["id"]} | CLOSED: {closed_text} | OPEN: {open_text}',
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
        cv2.putText(preview, "SPACE pause | 1-9 select | C closed | O open | Q quit",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
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
            d = doors[selected]
            if key == ord("c"):
                d["closed_ref"] = current.copy()
                cv2.imwrite(f'{d["id"]}_closed.jpg', current)
                print(f'{d["id"]}: CLOSED reference saved.')
            else:
                d["open_ref"] = current.copy()
                cv2.imwrite(f'{d["id"]}_open.jpg', current)
                print(f'{d["id"]}: OPEN reference saved.')
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Stopped during calibration.")
    cv2.destroyWindow("Door reference calibration")



def save_door_configuration(doors, frame_shape):
    """
    Save only the selected polygons and the exact OPEN/CLOSED calibration
    frames. Detection thresholds and comparison logic are not modified.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    saved_doors = []

    for door in doors:
        open_name = f'{door["id"]}_open.png'
        closed_name = f'{door["id"]}_closed.png'

        if not cv2.imwrite(str(CONFIG_DIR / open_name), door["open_ref"]):
            raise RuntimeError(f'Could not save {open_name}')
        if not cv2.imwrite(str(CONFIG_DIR / closed_name), door["closed_ref"]):
            raise RuntimeError(f'Could not save {closed_name}')

        saved_doors.append({
            "id": door["id"],
            "points": [[int(x), int(y)] for x, y in door["points"]],
            "open_ref": open_name,
            "closed_ref": closed_name
        })

    config = {
        "video_path": VIDEO_PATH,
        "frame_width": int(frame_shape[1]),
        "frame_height": int(frame_shape[0]),
        "doors": saved_doors
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"Door setup saved to: {CONFIG_FILE}")


def load_door_configuration(frame_shape):
    """Load previously saved polygons and calibration images."""
    if not CONFIG_FILE.exists():
        return None

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)

        expected_h, expected_w = frame_shape
        if (config.get("frame_width") != expected_w or
                config.get("frame_height") != expected_h):
            print("Saved setup resolution differs from this video.")
            print("A new setup is required.")
            return None

        doors = []
        for saved in config["doors"]:
            points = [(int(x), int(y)) for x, y in saved["points"]]
            if len(points) < 3:
                raise ValueError(f'{saved.get("id", "Door")} has invalid polygon.')

            open_path = CONFIG_DIR / saved["open_ref"]
            closed_path = CONFIG_DIR / saved["closed_ref"]

            # IMREAD_COLOR preserves the same BGR format used by calibration.
            open_ref = cv2.imread(str(open_path), cv2.IMREAD_COLOR)
            closed_ref = cv2.imread(str(closed_path), cv2.IMREAD_COLOR)

            if open_ref is None or closed_ref is None:
                raise FileNotFoundError(
                    f'Calibration image missing for {saved["id"]}.'
                )

            if (open_ref.shape[:2] != frame_shape or
                    closed_ref.shape[:2] != frame_shape):
                raise ValueError(
                    f'Calibration image size mismatch for {saved["id"]}.'
                )

            polygon = np.array(points, dtype=np.int32)
            mask = np.zeros(frame_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)

            doors.append({
                "id": saved["id"],
                "points": points,
                "mask": mask,
                "bbox": cv2.boundingRect(polygon),
                "open_ref": open_ref,
                "closed_ref": closed_ref,
                "state": "UNKNOWN",
                "candidate": None,
                "stable_count": 0
            })

        if not doors:
            return None

        print(f"Loaded {len(doors)} door(s) from {CONFIG_FILE}.")
        print("Using saved polygons and OPEN/CLOSED reference images.")
        return doors

    except Exception as exc:
        print(f"Could not load saved door setup: {exc}")
        print("A new setup is required.")
        return None


def main():
    ctx, queue, program = init_amd()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {VIDEO_PATH}. Keep it beside this script.")
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Cannot read the first video frame.")

    h, w = first_frame.shape[:2]

    parser = argparse.ArgumentParser(
        description="AMD OpenCL Shop Door Detection"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Force door marking and OPEN/CLOSED calibration again"
    )
    args = parser.parse_args()

    doors = None if args.setup else load_door_configuration((h, w))

    if doors is None:
        print("\nFirst-time setup / manual recalibration.")
        doors = select_doors(cap, first_frame)
        calibrate(cap, first_frame, doors)
        save_door_configuration(doors, (h, w))
    else:
        print("Saved setup loaded. Skipping door marking and calibration.")

    comparator = AMDComparator(ctx, queue, program, (h, w))
    print("\nUploading masks and references to AMD GPU...")
    for door in doors:
        comparator.add_door(door)
        print(door["id"], "ready on AMD GPU")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("\nAMD GPU monitoring started. GREEN=CLOSED, RED=OPEN. Press Q to stop.")
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

        comparator.upload_frame(frame)
        display = frame.copy()

        for door in doors:
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

            required_frames = (OPEN_STABLE_FRAMES if door["candidate"] == "OPEN"
                               else CLOSED_STABLE_FRAMES)
            if (door["candidate"] is not None and
                    door["stable_count"] >= required_frames and
                    door["state"] != door["candidate"]):
                old_state = door["state"]
                door["state"] = door["candidate"]
                save_event(door["id"], old_state, door["state"],
                           open_score, closed_score)
                play_alert(door["state"])

            display = draw_status(display, door)
            if SHOW_MATCH_SCORES:
                x, y, _, _ = door["bbox"]
                score_label = (f'Open:{open_score:.1f} Closed:{closed_score:.1f} '
                               f'Detected:{detected}')
                cv2.putText(display, score_label,
                            (max(4, x), min(h - 8, y + 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                            (255, 255, 255), 1, cv2.LINE_AA)

        processed += 1
        fps_count += 1
        now = time.perf_counter()
        if now - fps_start >= 1.0:
            shown_fps = fps_count / (now - fps_start)
            fps_count, fps_start = 0, now

        cv2.putText(display, f"AMD OPENCL | Processing FPS: {shown_fps:.1f}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)
        cv2.putText(display, "GREEN=CLOSED | RED=OPEN | ORANGE=UNCERTAIN | Q=QUIT",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)

        shown = display
        if DISPLAY_SCALE != 1.0:
            shown = cv2.resize(display, None, fx=DISPLAY_SCALE,
                               fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
        cv2.imshow("AMD OpenCL - Door Detection", shown)
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
