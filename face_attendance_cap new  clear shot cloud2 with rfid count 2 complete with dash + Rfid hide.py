import os
import cv2
import time
import requests
import tempfile
import numpy as np
import face_recognition
from datetime import datetime, date
from dotenv import load_dotenv
from supabase import create_client
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

# ===== Config =====
SNAPSHOT_BUCKET = "snapshots"
MATCH_THRESHOLD = 0.50          # face match threshold (lower is stricter)
UNKNOWN_FACE_THRESHOLD = 0.50   # consider same unknown if distance < this (dedupe)
SCAN_PHASE = 5                  # 0–5 sec: blue "SCANNING..."
DECISION_PHASE = 10             # 6–10 sec: decide (match / not match)
CAPTURE_DURATION = 15           # absolute max safety window

# ===== Helpers =====
def today_str():
    return date.today().isoformat()

def now_time_str():
    return datetime.now().strftime("%H:%M:%S")

def is_good_shot(face_bgr):
    if face_bgr.size == 0:
        return False
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    h, w = face_bgr.shape[:2]
    return (30 <= brightness <= 220) and (sharpness >= 20) and (h >= 60 and w >= 60)

def masked_face_encoding(img):
    face_locations = face_recognition.face_locations(img)
    if not face_locations:
        return None
    (top, right, bottom, left) = face_locations[0]
    masked_img = img[top:int(top + 0.6 * (bottom - top)), left:right]
    encs = face_recognition.face_encodings(masked_img)
    return encs[0] if encs else None

# ===== Supabase =====
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== Load students =====
print("⬇️ Loading students from Supabase...")
student_by_rfid = {}
encodings_by_rfid = {}

students = supabase.table("students").select("id,name,photo_url,RFID_code").execute().data or []
for s in students:
    name, sid, rfid, url_field = s.get("name"), s.get("id"), s.get("RFID_code"), s.get("photo_url")
    if not name or not sid or not url_field or not rfid:
        continue
    urls = url_field if isinstance(url_field, list) else [url_field]
    all_encs = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(resp.content)
                tmp.flush()
                img = face_recognition.load_image_file(tmp.name)
            encs = face_recognition.face_encodings(img)
            if not encs:
                enc = masked_face_encoding(img)
                if enc is not None:
                    encs = [enc]
            if encs:
                all_encs.extend(encs)
        except Exception as e:
            print(f"❌ Failed to load photo for {name}: {e}")

    if all_encs:
        student_by_rfid[str(rfid)] = s
        encodings_by_rfid[str(rfid)] = all_encs
        print(f"✅ Loaded {name} with {len(all_encs)} encodings (RFID: {rfid})")
    else:
        print(f"⚠️ No usable faces for {name} (RFID: {rfid})")

# ===== Tkinter App =====
class RFIDFaceApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Smart Attendance System")
        self.master.geometry("1000x700")

        # Camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Cannot open camera")
            raise SystemExit(1)

        # UI
        self.video_label = tk.Label(self.master)
        self.video_label.pack(padx=10, pady=10)

        # Status labels
        self.rfid_status_var = tk.StringVar(value="🔄 Waiting for RFID card...")
        self.status_var = tk.StringVar(value="Idle. Scan RFID to start verification.")

        tk.Label(self.master, textvariable=self.rfid_status_var, font=("Arial", 14), fg="blue").pack(pady=5)
        tk.Label(self.master, textvariable=self.status_var, font=("Arial", 16, "bold")).pack(pady=5)

        rfid_frame = tk.Frame(self.master)
        rfid_frame.pack(pady=6)
        tk.Label(rfid_frame, text="Scan RFID:", font=("Arial", 12)).pack(side=tk.LEFT, padx=6)

        # 🔧 Hidden RFID input (password style)
        self.rfid_input = tk.Entry(rfid_frame, font=("Arial", 13), width=28, show="*")
        self.rfid_input.pack(side=tk.LEFT)
        self.rfid_input.bind("<Return>", self.on_rfid_enter)

        tk.Label(self.master, text="Attendance Log:", font=("Arial", 12, "bold")).pack(pady=5)
        self.log_area = scrolledtext.ScrolledText(self.master, width=120, height=12, font=("Consolas", 10))
        self.log_area.pack(padx=10, pady=5)
        self.log_area.config(state=tk.DISABLED)

        # Session state
        self.pending_rfid = None
        self.session_start_ts = 0.0
        self.session_attendance_marked = False
        self.session_match_logged = False
        self.session_unknown_logged = False
        self.last_boxes_and_labels = []

        # Unknown dedupe
        self.unknown_encodings = []

        self.update_frame()

    # ---------- Logging ----------
    def log(self, message):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, f"[{now_time_str()}] {message}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state=tk.DISABLED)
        print(message)

    # ---------- RFID ----------
    def on_rfid_enter(self, event=None):
        code = self.rfid_input.get().strip()
        self.rfid_input.delete(0, tk.END)
        self.reset_session()
        self.pending_rfid = code
        self.session_start_ts = time.time()

        if code in student_by_rfid:
            name = student_by_rfid[code]["name"]
            self.rfid_status_var.set(f"✅ RFID scanned successfully: {name}")
            self.status_var.set("📷 Please look at the camera...")
            self.log(f"🔑 RFID scanned successfully: {code} → {name}")
        else:
            self.rfid_status_var.set("❌ Unknown RFID card scanned!")
            self.status_var.set("⚠️ Capturing face for review...")
            self.log(f"🔑 Unknown RFID scanned: {code}")

    def reset_session(self):
        self.pending_rfid = None
        self.session_start_ts = 0.0
        self.session_attendance_marked = False
        self.session_match_logged = False
        self.session_unknown_logged = False
        self.last_boxes_and_labels = []

        # Reset UI
        self.rfid_status_var.set("🔄 Waiting for RFID card...")
        self.status_var.set("Idle. Scan RFID to start verification.")
        self.rfid_input.delete(0, tk.END)
        self.rfid_input.focus_set()

    # ---------- Attendance ----------
    def mark_attendance_once(self, rfid_code, snapshot_url=None):
        if rfid_code not in student_by_rfid:
            return "error", "❌ No student found for this RFID."
        student = student_by_rfid[rfid_code]
        sid = student["id"]
        try:
            existing = supabase.table("attendance").select("id") \
                .eq("student_id", sid).eq("date", today_str()).limit(1).execute()
            if existing.data:
                return "already", f"ℹ️ Attendance already marked today for {student['name']}."
            supabase.table("attendance").insert({
                "student_id": sid,
                "date": today_str(),
                "time": now_time_str(),
                "snapshot_url": snapshot_url
            }).execute()
            return "marked", f"✅ Attendance marked for {student['name']}."
        except Exception as e:
            return "error", f"❌ Failed to mark attendance: {e}"

    # ---------- Storage ----------
    def upload_snapshot(self, face_bgr, prefix="unknown"):
        if face_bgr is None or face_bgr.size == 0:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{prefix}_{ts}.jpg"
        storage_path = f"{prefix}/{fname}"
        tmp_path = os.path.join(tempfile.gettempdir(), fname)
        cv2.imwrite(tmp_path, face_bgr)
        try:
            with open(tmp_path, "rb") as fh:
                supabase.storage.from_(SNAPSHOT_BUCKET).upload(
                    storage_path, fh, {"content-type": "image/jpeg", "upsert": "true"}
                )
            return supabase.storage.from_(SNAPSHOT_BUCKET).get_public_url(storage_path)
        except Exception as e:
            self.log(f"❌ Snapshot upload error: {e}")
            return None
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    def maybe_save_unknown_once(self, face_bgr, enc, rfid_code=None):
        if self.session_unknown_logged:
            return False, None, None
        if not is_good_shot(face_bgr):
            return False, "⚠️ Skipped bad quality face.", None
        if self.unknown_encodings:
            distances = face_recognition.face_distance(self.unknown_encodings, enc)
            if len(distances) and float(np.min(distances)) < UNKNOWN_FACE_THRESHOLD:
                self.session_unknown_logged = True
                return True, "ℹ️ Similar unknown face already saved earlier.", None
        prefix = "mismatch" if rfid_code in student_by_rfid else "unknown"
        url = self.upload_snapshot(face_bgr, prefix=prefix)
        if url:
            self.unknown_encodings.append(enc)
            self.session_unknown_logged = True
            return True, f"📸 Unknown face saved: {url}", url
        self.session_unknown_logged = True
        return False, "❌ Failed to save unknown face.", None

    # ---------- Frame loop ----------
    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.master.after(30, self.update_frame)
            return

        display = frame.copy()
        now_ts = time.time()
        active = (self.session_start_ts > 0)

        if active:
            elapsed = now_ts - self.session_start_ts
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            new_boxes_and_labels = []

            for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
                t, r, b, l = top * 4, right * 4, bottom * 4, left * 4
                face_crop = frame[t:b, l:r]

                match = False
                if self.pending_rfid in encodings_by_rfid:
                    ref_encs = encodings_by_rfid[self.pending_rfid]
                    distances = face_recognition.face_distance(ref_encs, enc)
                    if len(distances) and float(np.min(distances)) <= MATCH_THRESHOLD:
                        match = True

                # ----- Timing / UI logic -----
                if elapsed <= SCAN_PHASE:
                    color = (255, 0, 0); label_text = "SCANNING..."
                    self.status_var.set("Scanning… Please hold still.")
                elif elapsed <= DECISION_PHASE:
                    if self.pending_rfid in student_by_rfid:
                        if match:
                            color = (0, 255, 0)
                            label_text = f"VERIFIED: {student_by_rfid[self.pending_rfid]['name']}"
                            self.status_var.set("Verified ✅")
                            if not self.session_attendance_marked and not self.session_match_logged:
                                snapshot_url = self.upload_snapshot(face_crop, prefix="attendance")
                                status, msg = self.mark_attendance_once(self.pending_rfid, snapshot_url=snapshot_url)
                                self.session_attendance_marked = (status in ["marked", "already"])
                                self.log(msg)
                                self.session_match_logged = True

                        else:
                            color = (0, 0, 255); label_text = "MISMATCH"
                            self.status_var.set("Mismatch ❌")
                            if not self.session_unknown_logged:
                                saved, umsg, url = self.maybe_save_unknown_once(face_crop, enc, rfid_code=self.pending_rfid)
                                if umsg: self.log(umsg)
                    else:
                        color = (0, 0, 255); label_text = "UNKNOWN RFID"
                        self.status_var.set("Unknown RFID ❌")
                        if not self.session_unknown_logged:
                            saved, umsg, _ = self.maybe_save_unknown_once(face_crop, enc)
                            if umsg: self.log(umsg)
                else:
                    self.log("⏱ Session timeout → resetting.")
                    self.reset_session()
                    self.last_boxes_and_labels = []
                    self.render_frame(display)
                    self.master.after(30, self.update_frame)
                    return

                new_boxes_and_labels.append((l, t, r, b, color, label_text))

            self.last_boxes_and_labels = new_boxes_and_labels

        # Draw rectangles/labels
        for (l, t, r, b, color, text) in self.last_boxes_and_labels:
            cv2.rectangle(display, (l, t), (r, b), color, 2)
            cv2.putText(display, text, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        self.render_frame(display)
        self.master.after(30, self.update_frame)

    def render_frame(self, bgr):
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

# ===== Run App =====
if __name__ == "__main__":
    root = tk.Tk()
    app = RFIDFaceApp(root)
    root.mainloop()
