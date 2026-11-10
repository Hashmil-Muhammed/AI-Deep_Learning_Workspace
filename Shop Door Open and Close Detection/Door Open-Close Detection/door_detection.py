import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime

# ============================================================
# GPU + VIDEO SPEED SETTINGS
# ============================================================

VIDEO_PATH = "Footage.mp4"
LOG_FILE = "door_events.csv"

# GPU backend:
# "auto" = use NVIDIA CUDA through PyTorch if available, otherwise CPU.
# "cpu"  = force CPU.
PROCESSING_DEVICE = "auto"

# Speed controls:
# 1 = process every frame
# 2 = process every 2nd frame (faster, may miss very brief changes)
# 3 = process every 3rd frame (faster still, less accurate for brief events)
FRAME_SKIP = 2

# Display:
DISPLAY_SCALE = 0.85       # 1.0 = original size; smaller can improve display speed
WAIT_KEY_MS = 1            # 1 ms instead of 30 ms
SHOW_DIAGNOSTICS = True

# Door detection settings (same matching approach as your original code)
STABLE_FRAMES = 8
MAX_MATCH_DISTANCE = 35.0
MIN_MATCH_GAP = 2.0
FILL_ALPHA = 0.35

# ============================================================
# OPTIONAL NVIDIA GPU BACKEND
# ============================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

GPU_ENABLED = (
    PROCESSING_DEVICE.lower() != "cpu"
    and TORCH_AVAILABLE
    and torch.cuda.is_available()
)

if GPU_ENABLED:
    DEVICE = torch.device("cuda")
    print(f"[GPU] NVIDIA CUDA enabled: {torch.cuda.get_device_name(0)}")
    print("[GPU] Door image comparisons will run using PyTorch CUDA.")
else:
    DEVICE = None
    print("[CPU] CUDA/PyTorch GPU unavailable or CPU forced.")
    print("[CPU] Using OpenCV + NumPy CPU processing.")

# ============================================================
# ALERT
# ============================================================

try:
    import winsound

    def play_alert(state):
        winsound.Beep(1500 if state == "OPEN" else 900, 450)
except ImportError:
    def play_alert(state):
        print("\a", end="", flush=True)

# ============================================================
# IMAGE PROCESSING
# ============================================================

def make_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def prepare_reference(frame, mask):
    """
    Prepare reference grayscale image once.
    CPU path: NumPy array.
    GPU path: grayscale image and mask copied to CUDA once.
    """
    gray = make_gray(frame)

    if GPU_ENABLED:
        gray_gpu = torch.from_numpy(gray).to(
            DEVICE, dtype=torch.float32, non_blocking=False
        )
        mask_gpu = torch.from_numpy(mask > 0).to(DEVICE)
        return {
            "gray_cpu": gray,
            "gray_gpu": gray_gpu,
            "mask_gpu": mask_gpu
        }

    return {
        "gray_cpu": gray,
        "gray_gpu": None,
        "mask_gpu": None
    }


def masked_difference_prepared(frame_gray, reference, mask):
    """Mean absolute pixel difference inside one door polygon."""
    if GPU_ENABLED:
        # Upload current grayscale frame once per processed frame.
        # This function receives a CPU gray frame; GPU transfer is handled
        # by the caller through the cached frame tensor.
        current_gpu = torch.from_numpy(frame_gray).to(
            DEVICE, dtype=torch.float32, non_blocking=False
        )
        diff = torch.abs(current_gpu - reference["gray_gpu"])
        values = diff[reference["mask_gpu"]]
        if values.numel() == 0:
            return 999.0
        return float(values.mean().item())

    pixels = cv2.absdiff(frame_gray, reference["gray_cpu"])[mask > 0]
    if pixels.size == 0:
        return 999.0
    return float(np.mean(pixels))


def prepare_reference_for_door(frame, mask):
    gray = make_gray(frame)
    if GPU_ENABLED:
        return {
            "gray": torch.from_numpy(gray).to(DEVICE, dtype=torch.float32),
            "mask": torch.from_numpy(mask > 0).to(DEVICE)
        }
    return {"gray": gray, "mask": mask > 0}


def compare_current_to_reference(current_gray, reference, current_gpu=None):
    """Compare using a frame already uploaded to GPU, or use CPU NumPy."""
    if GPU_ENABLED:
        diff = torch.abs(current_gpu - reference["gray"])
        vals = diff[reference["mask"]]
        if vals.numel() == 0:
            return 999.0
        return float(torch.mean(vals).item())

    vals = cv2.absdiff(current_gray, reference["gray"])[reference["mask"]]
    if vals.size == 0:
        return 999.0
    return float(np.mean(vals))


# ============================================================
# LOGGING
# ============================================================

def save_event(door_id, old_state, new_state, open_score, closed_score):
    now = datetime.now()
    exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "Date", "Time", "Door ID", "Previous State",
                "New State", "Open Match Score", "Closed Match Score"
            ])
        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            door_id,
            old_state,
            new_state,
            f"{open_score:.2f}",
            f"{closed_score:.2f}"
        ])

    print(f"{now:%H:%M:%S} | {door_id}: {old_state} -> {new_state}")


# ============================================================
# DISPLAY
# ============================================================

def draw_status(frame, door):
    """Fill inside the door polygon: green=closed, red=open."""
    state = door["state"]

    if state == "CLOSED":
        color = (0, 200, 0)
    elif state == "OPEN":
        color = (0, 0, 255)
    else:
        color = (0, 165, 255)

    polygon = np.array(door["points"], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon], color)
    frame = cv2.addWeighted(
        overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0
    )
    cv2.polylines(frame, [polygon], True, color, 3)

    x, y, w, h = door["bbox"]
    label = f'{door["id"]}  {state}'
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
    )
    label_y = max(th + 12, y - 7)

    cv2.rectangle(
        frame, (x, label_y - th - 9), (x + tw + 12, label_y + 5),
        (25, 25, 25), -1
    )
    cv2.putText(
        frame, label, (x + 6, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2
    )

    if SHOW_DIAGNOSTICS:
        cv2.putText(
            frame,
            f'O:{door["open_score"]:.1f} C:{door["closed_score"]:.1f}',
            (x, min(frame.shape[0] - 10, y + h + 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1
        )

    return frame


# ============================================================
# OPEN VIDEO
# ============================================================

# Try hardware-accelerated video decoding where the OpenCV build supports it.
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
if not cap.isOpened():
    cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit("Cannot open Footage.mp4. Keep it beside this script.")

# Ask OpenCV to use hardware acceleration if available in this build/backend.
try:
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
except Exception:
    pass

ok, first_frame = cap.read()
if not ok:
    raise SystemExit("Cannot read the first video frame.")

frame_h, frame_w = first_frame.shape[:2]
doors = []

# ============================================================
# DRAW SEPARATE DOOR MASKS
# ============================================================

print("\nMARK EACH DOOR")
print("Draw a polygon around the FULL DOOR SWING AREA / DOORWAY.")
print("Include where the door panel appears when CLOSED and OPEN.")
print("Click polygon points, press ENTER to save this door.")
print("Then type y in terminal to add another door, or n to finish.")
print("R = reset current points; ESC = quit.")

while True:
    points = []
    window = f"Mark Door {len(doors) + 1}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.setMouseCallback(window, mouse_callback)

    while True:
        preview = first_frame.copy()
        for p in points:
            cv2.circle(preview, p, 5, (0, 255, 255), -1)
        if len(points) > 1:
            cv2.polylines(
                preview, [np.array(points, np.int32)],
                False, (0, 255, 255), 2
            )
        cv2.putText(
            preview, "Mark ONE door swing area | ENTER=save | R=reset",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (0, 255, 255), 2
        )
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            break
        elif key == ord("r"):
            points = []
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")

    cv2.destroyWindow(window)

    if len(points) < 3:
        print("Please select at least 3 points.")
        continue

    polygon = np.array(points, dtype=np.int32)
    full_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.fillPoly(full_mask, [polygon], 255)

    doors.append({
        "id": f"DOOR_{len(doors)+1:02d}",
        "points": points,
        "mask": full_mask,
        "bbox": cv2.boundingRect(polygon),
        "open_ref": None,
        "closed_ref": None,
        "state": "UNKNOWN",
        "candidate": None,
        "stable_count": 0,
        "open_score": 999.0,
        "closed_score": 999.0
    })

    print(f'{doors[-1]["id"]} mask saved.')
    if input("Mark another door? (y/n): ").strip().lower() != "y":
        break

if not doors:
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit("No door masks selected.")

# ============================================================
# CALIBRATE OPEN AND CLOSED REFERENCES
# ============================================================

print("\nCALIBRATION")
print("SPACE = pause/resume video")
print("1-9 = select a door")
print("C = save selected door's CLOSED reference")
print("O = save selected door's OPEN reference")
print("Calibrate BOTH states for EVERY door. Q = quit.")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
paused = False
current_frame = first_frame.copy()
selected_index = 0


def all_references_saved():
    return all(
        d["open_ref"] is not None and d["closed_ref"] is not None
        for d in doors
    )


while not all_references_saved():
    if not paused:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        current_frame = frame

    preview = current_frame.copy()
    for i, door in enumerate(doors):
        colour = (0, 255, 255) if i == selected_index else (190, 190, 190)
        cv2.polylines(
            preview, [np.array(door["points"], np.int32)],
            True, colour, 2
        )
        bx, by, _, _ = door["bbox"]
        cv2.putText(
            preview, door["id"], (bx, max(22, by - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2
        )

    selected = doors[selected_index]
    closed_text = "SAVED" if selected["closed_ref"] is not None else "NEEDED"
    open_text = "SAVED" if selected["open_ref"] is not None else "NEEDED"

    cv2.putText(
        preview,
        f'{selected["id"]} | CLOSED: {closed_text} | OPEN: {open_text}',
        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2
    )
    cv2.putText(
        preview, "SPACE pause | 1-9 select | C closed | O open | Q quit",
        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2
    )
    cv2.imshow("Door reference calibration", preview)

    key = cv2.waitKey(30 if not paused else 0) & 0xFF

    if key == ord(" "):
        paused = not paused
    elif ord("1") <= key <= ord("9"):
        idx = key - ord("1")
        if idx < len(doors):
            selected_index = idx
    elif key in (ord("c"), ord("o")):
        if not paused:
            print("Pause the video first, then press C or O.")
            continue

        door = doors[selected_index]
        if key == ord("c"):
            door["closed_ref"] = current_frame.copy()
            cv2.imwrite(f'{door["id"]}_closed.jpg', current_frame)
            print(f'{door["id"]}: CLOSED reference saved.')
        else:
            door["open_ref"] = current_frame.copy()
            cv2.imwrite(f'{door["id"]}_open.jpg', current_frame)
            print(f'{door["id"]}: OPEN reference saved.')
    elif key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit("Stopped during calibration.")

cv2.destroyWindow("Door reference calibration")

# Prepare both reference images ONCE, instead of repeatedly converting them.
for door in doors:
    door["open_ref_prepared"] = prepare_reference_for_door(
        door["open_ref"], door["mask"]
    )
    door["closed_ref_prepared"] = prepare_reference_for_door(
        door["closed_ref"], door["mask"]
    )

# ============================================================
# MONITOR ALL DOORS
# ============================================================

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("\nMonitoring started: GREEN = CLOSED, RED = OPEN.")
print(f"Processing every {FRAME_SKIP} frame(s). Press Q to stop.")

frame_number = 0
last_display = first_frame.copy()
fps_start = time.time()
processed_count = 0
display_fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Video finished.")
        break

    frame_number += 1

    # Skip frames to speed up processing/playback.
    if FRAME_SKIP > 1 and (frame_number - 1) % FRAME_SKIP != 0:
        continue

    # Convert the current frame to grayscale once.
    current_gray = make_gray(frame)

    # Upload the current grayscale frame only ONCE per processed frame.
    current_gpu = None
    if GPU_ENABLED:
        current_gpu = torch.from_numpy(current_gray).to(
            DEVICE, dtype=torch.float32, non_blocking=False
        )

    display = frame.copy()

    for door in doors:
        open_score = compare_current_to_reference(
            current_gray, door["open_ref_prepared"], current_gpu
        )
        closed_score = compare_current_to_reference(
            current_gray, door["closed_ref_prepared"], current_gpu
        )

        door["open_score"] = open_score
        door["closed_score"] = closed_score

        best_score = min(open_score, closed_score)
        score_gap = abs(open_score - closed_score)

        if best_score > MAX_MATCH_DISTANCE or score_gap < MIN_MATCH_GAP:
            detected = "UNKNOWN"
        elif open_score < closed_score:
            detected = "OPEN"
        else:
            detected = "CLOSED"

        if detected == "UNKNOWN":
            door["candidate"] = None
            door["stable_count"] = 0
        elif detected == door["candidate"]:
            door["stable_count"] += 1
        else:
            door["candidate"] = detected
            door["stable_count"] = 1

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

    processed_count += 1
    elapsed = time.time() - fps_start
    if elapsed >= 1.0:
        display_fps = processed_count / elapsed
        processed_count = 0
        fps_start = time.time()

    cv2.putText(
        display,
        f"GREEN=CLOSED | RED=OPEN | ORANGE=UNCERTAIN | FPS={display_fps:.1f}",
        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (255, 255, 255), 2
    )

    if DISPLAY_SCALE != 1.0:
        display = cv2.resize(
            display, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA
        )

    last_display = display
    cv2.imshow("GPU Door Open-Close Detection", display)

    if cv2.waitKey(WAIT_KEY_MS) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped. Logs saved to", LOG_FILE)
