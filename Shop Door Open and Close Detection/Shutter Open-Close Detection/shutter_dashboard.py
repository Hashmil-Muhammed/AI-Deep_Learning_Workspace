"""
Premium Shop Shutter Monitoring Dashboard
Starts the detector silently and displays its annotated frames and filtered run log.
"""

import os
import re
import csv
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
from PIL import Image, ImageTk

# --------------------------- SETTINGS ---------------------------
BASE_DIR = Path(__file__).resolve().parent
DETECTOR_SCRIPT = "shutter_detection_amd_opencl_dashboard_mode.py"
CSV_FILE = "shutter_events_opencl.csv"
FRAME_FILE = "latest_detector_frame.jpg"
FRAME_POLL_MS = 35
LOG_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2}\s+\|\s+SHUTTER_\d+:"  # state transitions
    r"|^Video finished\."
    r"|^Stopped\. Frames processed:"
    r"|^Elapsed seconds:"
    r"|^Average processing FPS:"
    r"|^CSV log:"
)
# ---------------------------------------------------------------

BG = "#0B1220"
SURFACE = "#111C2E"
PANEL = "#162338"
PANEL_ALT = "#0D1728"
BORDER = "#263750"
TEXT = "#E6EDF7"
MUTED = "#8FA2BC"
ACCENT = "#60A5FA"
GREEN = "#22C55E"
RED = "#F05252"
ORANGE = "#F59E0B"
FONT = "Segoe UI"


class PremiumDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Shop Shutter Monitoring | Control Center")
        self.root.geometry("1500x900")
        self.root.minsize(1120, 700)
        self.root.configure(bg=BG)

        self.process = None
        self.photo = None
        self.preview_paused = False
        self.log_lines = deque(maxlen=500)
        self.log_lock = threading.Lock()
        self.last_frame_mtime = 0
        self.last_state = "WAITING"
        self.last_shutter = "SHUTTER_01"
        self.event_total = 0
        self.process_finished = False

        self._configure_styles()
        self._build_ui()
        self.root.after(400, self.start_detector)
        self.root.after(FRAME_POLL_MS, self.refresh_frame)
        self.root.after(250, self.refresh_logs)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Premium.Treeview",
            background=PANEL_ALT, foreground=TEXT, fieldbackground=PANEL_ALT,
            rowheight=30, borderwidth=0, font=(FONT, 9)
        )
        style.configure(
            "Premium.Treeview.Heading", background="#20324B",
            foreground="#C9D8EA", font=(FONT, 8, "bold"), relief="flat"
        )
        style.map("Premium.Treeview", background=[("selected", "#28466A")])

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=22, pady=(18, 14))
        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left")
        tk.Label(brand, text="SHUTTER", bg=BG, fg=ACCENT,
                 font=(FONT, 10, "bold")).pack(anchor="w")
        tk.Label(brand, text="Monitoring Control Center", bg=BG, fg=TEXT,
                 font=(FONT, 21, "bold")).pack(anchor="w", pady=(2, 0))
        self.status_dot = tk.Label(
            header, text="● INITIALIZING", bg=BG, fg=ORANGE,
            font=(FONT, 10, "bold")
        )
        self.status_dot.pack(side="right", padx=4)

        # Main split
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True, padx=18, pady=(0, 12))

        # Video panel
        left = tk.Frame(content, bg=PANEL, highlightbackground=BORDER,
                        highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self._section_heading(left, "LIVE DETECTION FEED", "ANNOTATED OUTPUT")
        self.video_box = tk.Frame(left, bg="#050A13")
        self.video_box.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.video_label = tk.Label(
            self.video_box, bg="#050A13", fg=MUTED,
            text="Waiting for the detector video feed…",
            font=(FONT, 13)
        )
        self.video_label.pack(fill="both", expand=True)
        self.video_meta = tk.Label(
            left, text="AMD OpenCL • Waiting for first frame",
            bg=PANEL, fg=MUTED, font=(FONT, 9), anchor="w"
        )
        self.video_meta.pack(fill="x", padx=14, pady=(0, 12))

        # Right panel
        right = tk.Frame(content, bg=PANEL, width=390,
                         highlightbackground=BORDER, highlightthickness=1)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self._section_heading(right, "DETECTION OVERVIEW", "LIVE STATUS")
        self.status_card = tk.Frame(right, bg=PANEL_ALT)
        self.status_card.pack(fill="x", padx=12, pady=(0, 8))
        self.state_title = tk.Label(
            self.status_card, text="WAITING FOR DETECTION",
            bg=PANEL_ALT, fg=ORANGE, font=(FONT, 16, "bold"),
            pady=18, wraplength=330
        )
        self.state_title.pack(fill="x")
        self.state_subtitle = tk.Label(
            self.status_card, text="No state transition received yet",
            bg=PANEL_ALT, fg=MUTED, font=(FONT, 9), pady=0
        )
        self.state_subtitle.pack(pady=(0, 13))

        stats = tk.Frame(right, bg=PANEL)
        stats.pack(fill="x", padx=12, pady=(3, 12))
        self._metric(stats, "TOTAL EVENTS", "0", "events")
        self._metric(stats, "LATEST SHUTTER", "—", "identifier")

        # Log terminal panel
        self._section_heading(right, "DETECTOR RUN LOG", "EVENTS & SUMMARY")
        terminal = tk.Frame(right, bg="#050B15",
                            highlightbackground="#26364D", highlightthickness=1)
        terminal.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        terminal_top = tk.Frame(terminal, bg="#101B2B")
        terminal_top.pack(fill="x")
        tk.Label(terminal_top, text="●  LIVE OUTPUT",
                 bg="#101B2B", fg=GREEN, font=(FONT, 8, "bold")
                 ).pack(side="left", padx=10, pady=8)
        tk.Label(terminal_top, text="FILTERED",
                 bg="#101B2B", fg=MUTED, font=(FONT, 8)
                 ).pack(side="right", padx=10)
        self.log_text = tk.Text(
            terminal, bg="#050B15", fg="#B9C9DD", insertbackground=TEXT,
            relief="flat", wrap="word", font=("Consolas", 9),
            padx=10, pady=10, state="disabled", spacing1=2, spacing3=3
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(terminal, orient="vertical",
                                   command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.tag_configure("event_open", foreground="#4ADE80")
        self.log_text.tag_configure("event_closed", foreground="#F87171")
        self.log_text.tag_configure("summary", foreground="#93C5FD")
        self.log_text.tag_configure("normal", foreground="#C5D2E3")

        # Footer controls
        controls = tk.Frame(right, bg=PANEL)
        controls.pack(fill="x", padx=12, pady=(0, 12))
        self.pause_button = self._button(
            controls, "PAUSE PREVIEW", self.toggle_pause, "#263B57"
        )
        self.pause_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.stop_button = self._button(
            controls, "STOP DETECTOR", self.stop_detector, "#7F1D1D"
        )
        self.stop_button.pack(side="left", fill="x", expand=True, padx=(5, 0))

        footer = tk.Label(
            self.root,
            text="OPEN  ● GREEN     CLOSED  ● RED     UNKNOWN  ● AMBER"
                 "       |       AMD OPENCL SHUTTER ANALYTICS",
            bg=BG, fg=MUTED, font=(FONT, 9)
        )
        footer.pack(pady=(0, 10))

    def _section_heading(self, parent, title, tag):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", padx=14, pady=(13, 10))
        tk.Label(row, text=title, bg=PANEL, fg=TEXT,
                 font=(FONT, 9, "bold")).pack(side="left")
        tk.Label(row, text=tag, bg=PANEL, fg=ACCENT,
                 font=(FONT, 8, "bold")).pack(side="right")

    def _metric(self, parent, title, value, subtitle):
        box = tk.Frame(parent, bg=PANEL_ALT)
        box.pack(side="left", fill="x", expand=True, padx=2)
        tk.Label(box, text=title, bg=PANEL_ALT, fg=MUTED,
                 font=(FONT, 7, "bold")).pack(anchor="w", padx=9, pady=(8, 2))
        label = tk.Label(box, text=value, bg=PANEL_ALT, fg=TEXT,
                         font=(FONT, 13, "bold"))
        label.pack(anchor="w", padx=9)
        tk.Label(box, text=subtitle, bg=PANEL_ALT, fg=MUTED,
                 font=(FONT, 7)).pack(anchor="w", padx=9, pady=(0, 8))
        if title == "TOTAL EVENTS":
            self.events_value = label
        else:
            self.shutter_value = label

    def _button(self, parent, text, command, color):
        return tk.Button(
            parent, text=text, command=command, bg=color, fg="white",
            activebackground=color, activeforeground="white",
            relief="flat", bd=0, font=(FONT, 8, "bold"), pady=11,
            cursor="hand2"
        )

    def start_detector(self):
        script = BASE_DIR / DETECTOR_SCRIPT
        if not script.is_file():
            self._set_status("DETECTOR FILE NOT FOUND", RED)
            messagebox.showerror(
                "Detector not found",
                f"Cannot find {script}\n\nCheck DETECTOR_SCRIPT at the top of the dashboard."
            )
            return

        frame_path = BASE_DIR / FRAME_FILE
        try:
            if frame_path.exists():
                frame_path.unlink()
        except OSError:
            pass

        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(script)],
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags
            )
            threading.Thread(target=self._read_process_output, daemon=True).start()
            self._set_status("● DETECTOR RUNNING", GREEN)
        except Exception as exc:
            self._set_status("DETECTOR START FAILED", RED)
            messagebox.showerror("Start error", str(exc))

    def _read_process_output(self):
        if not self.process or not self.process.stdout:
            return
        for raw in self.process.stdout:
            line = raw.strip()
            if not line:
                continue
            if LOG_PATTERN.search(line):
                with self.log_lock:
                    self.log_lines.append(line)
                if "SHUTTER_" in line and "->" in line:
                    match = re.search(r"(SHUTTER_\d+):\s+([A-Z]+)\s+->\s+([A-Z]+)", line)
                    if match:
                        self.last_shutter = match.group(1)
                        self.last_state = match.group(3)
                        self.event_total += 1
            if line.startswith("Stopped. Frames processed:"):
                self.process_finished = True

    def refresh_logs(self):
        with self.log_lock:
            pending = list(self.log_lines)
            self.log_lines.clear()

        if pending:
            self.log_text.configure(state="normal")
            for line in pending:
                if "-> OPEN" in line:
                    tag = "event_open"
                elif "-> CLOSED" in line:
                    tag = "event_closed"
                elif line.startswith(("Video finished.", "Stopped.", "Elapsed seconds:",
                                      "Average processing FPS:", "CSV log:")):
                    tag = "summary"
                else:
                    tag = "normal"
                self.log_text.insert("end", line + "\n", tag)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        if self.event_total:
            color = GREEN if self.last_state == "OPEN" else RED if self.last_state == "CLOSED" else ORANGE
            self.state_title.config(text=f"{self.last_shutter}: {self.last_state}", fg=color)
            self.state_subtitle.config(text="Latest confirmed shutter transition")
        self.events_value.config(text=str(self.event_total))
        self.shutter_value.config(text=self.last_shutter if self.event_total else "—")

        if self.process and self.process.poll() is not None:
            self._set_status("● VIDEO FINISHED" if self.process_finished else "● DETECTOR STOPPED",
                             MUTED if self.process_finished else RED)
        self.root.after(250, self.refresh_logs)

    def refresh_frame(self):
        if not self.preview_paused:
            frame_path = BASE_DIR / FRAME_FILE
            try:
                if frame_path.exists():
                    mtime = frame_path.stat().st_mtime_ns
                    if mtime != self.last_frame_mtime:
                        # Read bytes safely because Windows may briefly lock the
                        # JPG while the detector atomically replaces it.
                        try:
                            frame_bytes = frame_path.read_bytes()
                        except OSError:
                            frame_bytes = None

                        frame = None
                        if frame_bytes:
                            import numpy as np
                            frame = cv2.imdecode(
                                np.frombuffer(frame_bytes, dtype=np.uint8),
                                cv2.IMREAD_COLOR
                            )

                        if frame is not None:
                            self.last_frame_mtime = mtime
                            box_w = max(320, self.video_box.winfo_width() - 8)
                            box_h = max(240, self.video_box.winfo_height() - 8)
                            h, w = frame.shape[:2]
                            scale = min(box_w / w, box_h / h)
                            resized = cv2.resize(
                                frame, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA
                            )
                            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                            self.video_label.config(image=self.photo, text="")
                            self.video_meta.config(
                                text=f"Annotated detector output  •  {w} × {h}  •  AMD OpenCL"
                            )
            except (OSError, cv2.error):
                pass
        self.root.after(FRAME_POLL_MS, self.refresh_frame)

    def _set_status(self, text, color):
        self.status_dot.config(text=text, fg=color)

    def toggle_pause(self):
        self.preview_paused = not self.preview_paused
        self.pause_button.config(
            text="RESUME PREVIEW" if self.preview_paused else "PAUSE PREVIEW"
        )

    def stop_detector(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self._set_status("● DETECTOR STOPPING", ORANGE)

    def close(self):
        self.stop_detector()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PremiumDashboard(root)
    root.mainloop()
