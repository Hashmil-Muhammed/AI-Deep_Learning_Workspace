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

# ============================================================
# AMD Radeon R5 M430 / OpenCL - SHOP SHUTTER DETECTION
# Put Footage1.mp4 in the same folder as this Python file.
# ============================================================

VIDEO_PATH = "Footage3.mp4"
FRAME_OUTPUT = "latest_detector_frame.jpg"
LOG_FILE = "shutter_events_opencl.csv"
CONFIG_DIR = Path("shutter_configuration")
CONFIG_FILE = CONFIG_DIR / "shutters.json"

OPEN_STABLE_FRAMES = 5
CLOSED_STABLE_FRAMES = 8
SHOW_MATCH_SCORES = True

# These are starting values. Tune them after checking real scores.
MAX_MATCH_DISTANCE = 35.0
MIN_MATCH_GAP = 2.0

FILL_ALPHA = 0.30
FRAME_SKIP = 1
DISPLAY_WAIT_MS = 1
DISPLAY_SCALE = 0.75
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
    """Find an AMD OpenCL GPU and compile the comparison kernel."""
    try:
        platforms = cl.get_platforms()
    except Exception as exc:
        raise RuntimeError(
            "OpenCL platform discovery failed. Check the AMD OpenCL driver "
            "and PyOpenCL installation."
        ) from exc

    for platform in platforms:
        platform_name = platform.name.upper()
        platform_vendor = getattr(platform, "vendor", "").upper()

        if "AMD" not in platform_name and "ADVANCED MICRO DEVICES" not in platform_vendor:
            continue

        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if not devices:
            continue

        device = devices[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        program = cl.Program(context, KERNEL).build()

        print("\nAMD OpenCL platform:", platform.name)
        print("Selected GPU:", device.name)
        print("Vendor:", device.vendor)
        print("OpenCL version:", device.version)
        return context, queue, program

    raise RuntimeError(
        "AMD OpenCL GPU not found. Confirm that the AMD OpenCL runtime "
        "is installed and that PyOpenCL can detect the Radeon GPU."
    )


class AMDComparator:
    """Uploads the shutter mask/references once and compares each frame on GPU."""

    def __init__(self, context, queue, program, frame_shape):
        self.context = context
        self.queue = queue
        self.height, self.width = frame_shape
        self.n = self.height * self.width

        self.groups = (self.n + LOCAL_SIZE - 1) // LOCAL_SIZE
        self.global_size = self.groups * LOCAL_SIZE

        mf = cl.mem_flags
        self.current_buffer = cl.Buffer(context, mf.READ_ONLY, self.n)
        self.partial_buffer = cl.Buffer(
            context, mf.WRITE_ONLY,
            self.groups * np.dtype(np.float32).itemsize
        )
        self.partial_host = np.empty(self.groups, dtype=np.float32)

        # Create the kernel once (avoids repeated-kernel-retrieval warnings).
        self.kernel = cl.Kernel(program, "masked_absdiff_reduce")
        self.gpu_shutters = {}

    def add_shutter(self, shutter):
        mf = cl.mem_flags
        mask = np.ascontiguousarray(shutter["mask"], dtype=np.uint8)
        open_ref = make_gray(shutter["open_ref"])
        closed_ref = make_gray(shutter["closed_ref"])

        if open_ref.shape != (self.height, self.width):
            raise ValueError(f'{shutter["id"]}: OPEN reference size mismatch.')
        if closed_ref.shape != (self.height, self.width):
            raise ValueError(f'{shutter["id"]}: CLOSED reference size mismatch.')

        pixel_count = int(np.count_nonzero(mask))
        if pixel_count == 0:
            raise ValueError(f'{shutter["id"]} has an empty ROI mask.')

        self.gpu_shutters[shutter["id"]] = {
            "mask": cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mask
            ),
            "open_ref": cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=open_ref
            ),
            "closed_ref": cl.Buffer(
                self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=closed_ref
            ),
            "pixel_count": pixel_count
        }

    def upload_frame(self, frame):
        gray = make_gray(frame)
        cl.enqueue_copy(self.queue, self.current_buffer, gray)
        return gray

    def difference(self, shutter_id, reference_key):
        item = self.gpu_shutters[shutter_id]

        self.kernel(
            self.queue,
            (self.global_size,),
            (LOCAL_SIZE,),
            self.current_buffer,
            item[reference_key],
            item["mask"],
            self.partial_buffer,
            np.uint32(self.n),
            cl.LocalMemory(LOCAL_SIZE * np.dtype(np.float32).itemsize)
        )

        cl.enqueue_copy(
            self.queue, self.partial_host, self.partial_buffer
        ).wait()

        return float(np.sum(self.partial_host)) / item["pixel_count"]


def save_event(shutter_id, old_state, new_state, open_score, closed_score):
    now = datetime.now()
    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Date", "Time", "Shutter ID", "Previous State", "New State",
                "Open Match Score", "Closed Match Score"
            ])

        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            shutter_id,
            old_state,
            new_state,
            f"{open_score:.2f}",
            f"{closed_score:.2f}"
        ])

    print(f"{now:%H:%M:%S} | {shutter_id}: {old_state} -> {new_state}")


def draw_status(frame, shutter):
    state = shutter["state"]
    color = (
    (0, 0, 255) if state == "CLOSED"  # RED
    else (0, 200, 0) if state == "OPEN"  # GREEN
    else (0, 165, 255)  # ORANGE
    )

    polygon = np.array(shutter["points"], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    frame = cv2.addWeighted(
        overlay, FILL_ALPHA, frame, 1.0 - FILL_ALPHA, 0
    )
    cv2.polylines(frame, [polygon], True, color, 3)

    x, y, _, _ = shutter["bbox"]
    label = f'{shutter["id"]}  {state}'
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    label_y = max(text_h + 12, y - 7)

    cv2.rectangle(
        frame,
        (x, label_y - text_h - 9),
        (x + text_w + 12, label_y + 5),
        (25, 25, 25),
        -1
    )
    cv2.putText(
        frame, label, (x + 6, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
    )
    return frame


def select_shutters(first_frame):
    """Mark the complete shutter travel area, not only the visible shutter."""
    height, width = first_frame.shape[:2]
    shutters = []

    print("\nMARK EACH SHUTTER ROI")
    print("Click polygon points around the FULL shutter movement area.")
    print("Include the opening from the top housing down to the closed position.")
    print("Press ENTER to save; R to reset; ESC to cancel.")
    print("After saving, type y to mark another shutter or n to continue.")

    while True:
        points = []
        window = f"Mark Shutter {len(shutters) + 1}"
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
                cv2.polylines(
                    preview, [np.array(points, np.int32)],
                    False, (0, 255, 255), 2
                )

            instruction = "Mark FULL shutter travel area | ENTER=save | R=reset | ESC=quit"
            cv2.putText(
                preview, instruction, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):
                break
            if key == ord("r"):
                points = []
            if key == 27:
                cv2.destroyAllWindows()
                raise SystemExit("Setup cancelled.")

        cv2.destroyWindow(window)

        if len(points) < 3:
            print("Please select at least 3 polygon points.")
            continue

        polygon = np.array(points, dtype=np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        shutter = {
            "id": f"SHUTTER_{len(shutters) + 1:02d}",
            "points": points,
            "mask": mask,
            "bbox": cv2.boundingRect(polygon),
            "open_ref": None,
            "closed_ref": None,
            "state": "UNKNOWN",
            "candidate": None,
            "stable_count": 0
        }
        shutters.append(shutter)
        print(f'{shutter["id"]} ROI saved.')

        if input("Mark another shutter? (y/n): ").strip().lower() != "y":
            break

    if not shutters:
        raise SystemExit("No shutter ROI was selected.")

    return shutters


def calibrate_shutters(cap, first_frame, shutters):
    """
    For a video file: play through the clip, pause at a fully closed/open frame,
    then press C/O. If the video ends before both references are captured,
    it loops back to the beginning.
    """
    print("\nSHUTTER CALIBRATION")
    print("SPACE = pause/resume | 1-9 = select shutter")
    print("C = save CLOSED reference | O = save OPEN reference | Q = quit")
    print("Capture fully open/closed positions, preferably with no person in ROI.")

    paused = False
    current = first_frame.copy()
    selected = 0

    while not all(
        shutter["open_ref"] is not None and shutter["closed_ref"] is not None
        for shutter in shutters
    ):
        if not paused:
            ok, frame = cap.read()
            if not ok:
                # This code is for a seekable video file, not a live RTSP stream.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Cannot read video during calibration.")
            current = frame

        preview = current.copy()

        for index, shutter in enumerate(shutters):
            color = (0, 255, 255) if index == selected else (190, 190, 190)
            cv2.polylines(
                preview, [np.array(shutter["points"], np.int32)],
                True, color, 2
            )
            bx, by, _, _ = shutter["bbox"]
            cv2.putText(
                preview, shutter["id"], (bx, max(22, by - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

        shutter = shutters[selected]
        closed_text = "SAVED" if shutter["closed_ref"] is not None else "NEEDED"
        open_text = "SAVED" if shutter["open_ref"] is not None else "NEEDED"

        cv2.putText(
            preview,
            f'{shutter["id"]} | CLOSED: {closed_text} | OPEN: {open_text}',
            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2
        )
        cv2.putText(
            preview,
            "SPACE pause | 1-9 select | C closed | O open | Q quit",
            (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
        )

        cv2.imshow("Shutter Reference Calibration", preview)
        key = cv2.waitKey(30 if not paused else 0) & 0xFF

        if key == ord(" "):
            paused = not paused

        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            if index < len(shutters):
                selected = index

        elif key in (ord("c"), ord("o")):
            if not paused:
                print("Pause the video first, then press C or O.")
                continue

            shutter = shutters[selected]
            if key == ord("c"):
                shutter["closed_ref"] = current.copy()
                print(f'{shutter["id"]}: CLOSED reference captured.')
            else:
                shutter["open_ref"] = current.copy()
                print(f'{shutter["id"]}: OPEN reference captured.')

        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Stopped during calibration.")

    cv2.destroyWindow("Shutter Reference Calibration")


def save_shutter_configuration(shutters, frame_shape):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    saved_shutters = []

    for shutter in shutters:
        open_name = f'{shutter["id"]}_open.png'
        closed_name = f'{shutter["id"]}_closed.png'

        if not cv2.imwrite(str(CONFIG_DIR / open_name), shutter["open_ref"]):
            raise RuntimeError(f"Could not save {open_name}")
        if not cv2.imwrite(str(CONFIG_DIR / closed_name), shutter["closed_ref"]):
            raise RuntimeError(f"Could not save {closed_name}")

        saved_shutters.append({
            "id": shutter["id"],
            "points": [[int(x), int(y)] for x, y in shutter["points"]],
            "open_ref": open_name,
            "closed_ref": closed_name
        })

    config = {
        "video_path": VIDEO_PATH,
        "frame_width": int(frame_shape[1]),
        "frame_height": int(frame_shape[0]),
        "shutters": saved_shutters
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    print(f"Shutter configuration saved to: {CONFIG_FILE}")


def load_shutter_configuration(frame_shape):
    if not CONFIG_FILE.exists():
        return None

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as file:
            config = json.load(file)

        expected_h, expected_w = frame_shape
        if (
            config.get("frame_width") != expected_w
            or config.get("frame_height") != expected_h
        ):
            print("Saved setup resolution differs from this video.")
            print("Run with --setup to create a new shutter setup.")
            return None

        shutters = []

        for saved in config["shutters"]:
            points = [(int(x), int(y)) for x, y in saved["points"]]
            if len(points) < 3:
                raise ValueError(f'{saved.get("id", "Shutter")} has invalid ROI.')

            open_path = CONFIG_DIR / saved["open_ref"]
            closed_path = CONFIG_DIR / saved["closed_ref"]

            open_ref = cv2.imread(str(open_path), cv2.IMREAD_COLOR)
            closed_ref = cv2.imread(str(closed_path), cv2.IMREAD_COLOR)

            if open_ref is None or closed_ref is None:
                raise FileNotFoundError(
                    f'Calibration image missing for {saved["id"]}.'
                )

            if open_ref.shape[:2] != frame_shape or closed_ref.shape[:2] != frame_shape:
                raise ValueError(f'Reference image size mismatch for {saved["id"]}.')

            polygon = np.array(points, dtype=np.int32)
            mask = np.zeros(frame_shape, dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)

            shutters.append({
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

        if not shutters:
            return None

        print(f"Loaded {len(shutters)} shutter(s) from {CONFIG_FILE}.")
        return shutters

    except Exception as exc:
        print(f"Could not load saved shutter setup: {exc}")
        print("A new setup is required.")
        return None


def classify_shutter(open_score, closed_score):
    best_score = min(open_score, closed_score)
    score_gap = abs(open_score - closed_score)

    if best_score > MAX_MATCH_DISTANCE or score_gap < MIN_MATCH_GAP:
        return "UNKNOWN"

    return "OPEN" if open_score < closed_score else "CLOSED"


def main():
    parser = argparse.ArgumentParser(
        description="AMD OpenCL Shop Shutter Detection"
    )
    parser.add_argument(
        "--video", default=VIDEO_PATH,
        help="Video filename (default: Footage1.mp4)"
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Force shutter ROI marking and OPEN/CLOSED calibration"
    )
    args = parser.parse_args()

    video_path = args.video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise SystemExit(
            f"Cannot open video: {video_path}. Put the video beside this script "
            "or pass its path using --video."
        )

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Cannot read the first video frame.")

    height, width = first_frame.shape[:2]
    frame_shape = (height, width)

    try:
        shutters = None if args.setup else load_shutter_configuration(frame_shape)

        if shutters is None:
            print("\nFirst-time setup / manual recalibration.")
            shutters = select_shutters(first_frame)
            calibrate_shutters(cap, first_frame, shutters)
            save_shutter_configuration(shutters, frame_shape)
        else:
            print("Saved shutter setup loaded. Skipping marking and calibration.")

        # Initialise AMD after setup so marking/calibration can be checked first.
        context, queue, program = init_amd()
        comparator = AMDComparator(context, queue, program, frame_shape)

        print("\nUploading shutter masks and references to AMD GPU...")
        for shutter in shutters:
            comparator.add_shutter(shutter)
            print(shutter["id"], "ready on AMD GPU")

        # Restart the video for monitoring.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("\nAMD GPU shutter monitoring started.")
        print("GREEN=CLOSED | RED=OPEN | ORANGE=UNKNOWN | Q=QUIT")

        frame_no = processed = fps_count = 0
        start_time = fps_start = time.perf_counter()
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

            for shutter in shutters:
                open_score = comparator.difference(shutter["id"], "open_ref")
                closed_score = comparator.difference(shutter["id"], "closed_ref")
                detected = classify_shutter(open_score, closed_score)

                if detected == "UNKNOWN":
                    shutter["candidate"] = None
                    shutter["stable_count"] = 0
                elif detected == shutter["candidate"]:
                    shutter["stable_count"] += 1
                else:
                    shutter["candidate"] = detected
                    shutter["stable_count"] = 1

                required_frames = (
                    OPEN_STABLE_FRAMES
                    if shutter["candidate"] == "OPEN"
                    else CLOSED_STABLE_FRAMES
                )

                if (
                    shutter["candidate"] is not None
                    and shutter["stable_count"] >= required_frames
                    and shutter["state"] != shutter["candidate"]
                ):
                    old_state = shutter["state"]
                    shutter["state"] = shutter["candidate"]
                    save_event(
                        shutter["id"], old_state, shutter["state"],
                        open_score, closed_score
                    )
                    play_alert(shutter["state"])

                display = draw_status(display, shutter)

                if SHOW_MATCH_SCORES:
                    x, y, _, _ = shutter["bbox"]
                    score_label = (
                        f'Open:{open_score:.1f} Closed:{closed_score:.1f} '
                        f'Detected:{detected}'
                    )
                    cv2.putText(
                        display, score_label,
                        (max(4, x), min(height - 8, y + 24)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (255, 255, 255), 1, cv2.LINE_AA
                    )

            processed += 1
            fps_count += 1
            now = time.perf_counter()

            if now - fps_start >= 1.0:
                shown_fps = fps_count / (now - fps_start)
                fps_count = 0
                fps_start = now

            cv2.putText(
                display, f"AMD OPENCL | Processing FPS: {shown_fps:.1f}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2
            )
            cv2.putText(
                display,
                "GREEN=CLOSED | RED=OPEN | ORANGE=UNKNOWN | Q=QUIT",
                (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2
            )

            shown = display
            if DISPLAY_SCALE != 1.0:
                shown = cv2.resize(
                    display, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA
                )

            success, encoded = cv2.imencode(".jpg", shown)

            if success:
                # Write to a temporary file first, then replace the dashboard frame.
                # Windows may briefly lock FRAME_OUTPUT while the dashboard reads it.
                temp_file = FRAME_OUTPUT + ".tmp"

                try:
                    with open(temp_file, "wb") as f:
                        f.write(encoded.tobytes())

                    try:
                        os.replace(temp_file, FRAME_OUTPUT)
                    except PermissionError:
                        # The dashboard may be reading the previous JPG at this moment.
                        # Skip this one update instead of stopping shutter detection.
                        try:
                            os.remove(temp_file)
                        except OSError:
                            pass

                except OSError as exc:
                    # Frame publishing is only for the dashboard; it must never
                    # interrupt the AMD OpenCL detection loop.
                    print(f"Dashboard frame update skipped: {exc}")
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except OSError:
                        pass


        elapsed = time.perf_counter() - start_time
        print("\nStopped. Frames processed:", processed)
        print("Elapsed seconds:", round(elapsed, 2))
        if elapsed > 0:
            print("Average processing FPS:", round(processed / elapsed, 2))
        print("CSV log:", LOG_FILE)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
