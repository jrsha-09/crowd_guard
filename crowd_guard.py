import cv2
import time
import os
from ultralytics import YOLO

# --- Firebase ---
import firebase_admin
from firebase_admin import credentials, db, storage

# --- Audio ---
from gtts import gTTS
import playsound
import threading

# === CONFIG ===
YOLO_MODEL_PATH = "yolo11s.pt"
VIDEO_PATH = r"C:\Users\goswa\Documents\Pillai\Shopping_1.mp4"
FIREBASE_KEY_PATH = r"C:\Users\goswa\Documents\Pillai\crowd.json"

INFER_EVERY_N_FRAMES = 5
UPLOAD_EVERY_N_FRAMES = 15
VIDEO_CHUNK_SECONDS = 30
ANNOUNCEMENT_INTERVAL = 30   # seconds between announcements

# --- Firebase setup ---
try:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://crowd-detection-acded-default-rtdb.firebaseio.com",
        "storageBucket": "crowd-detection-acded.firebasestorage.app"
    })
    bucket = storage.bucket()
    print("✅ Firebase initialized successfully.")
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    bucket = None

# --- Load YOLO ---
model = YOLO(YOLO_MODEL_PATH)
print(f"✅ YOLO model '{YOLO_MODEL_PATH}' loaded successfully.")


# === DENSITY LABELS ===
def get_density_label(count: int) -> str:
    if count > 10:
        return "High"
    elif count >= 5:
        return "Medium"
    else:
        return "Low"


# === AUDIO SYSTEM ===
last_announcement_time = 0

def play_announcement():
    """Plays multilingual announcements in sequence."""
    messages = {
        "en": "Please don't panic. Maintain a line to avoid inconvenience.",
        "hi": "कृपया घबराएँ नहीं। असुविधा से बचने के लिए एक पंक्ति बनाए रखें।",
        "mr": "कृपया घाबरू नका. गैरसोयी टाळण्यासाठी रांगेत उभे रहा."
    }

    for lang, text in messages.items():
        try:
            tts = gTTS(text=text, lang=lang)
            filename = f"announce_{lang}.mp3"
            tts.save(filename)
            playsound.playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"❌ Error playing {lang} announcement: {e}")


def maybe_trigger_announcement(zone_data):
    """Triggers announcement if any zone has Medium density and interval passed."""
    global last_announcement_time
    now = time.time()

    if any(z["density"] == "Medium" for z in zone_data.values()):
        if now - last_announcement_time > ANNOUNCEMENT_INTERVAL:
            last_announcement_time = now
            threading.Thread(target=play_announcement, daemon=True).start()
            print("📢 Announcement triggered!")


# === VIDEO UPLOAD ===
def upload_video_chunk(filename: str):
    if not bucket or not os.path.exists(filename):
        print(f"⚠ Skipping upload for {filename}")
        return
    try:
        blob = bucket.blob(f"videos/{filename}")
        blob.upload_from_filename(filename, content_type="video/mp4")
        blob.make_public()
        db.reference("/crowd/latest_video").set({
            "url": blob.public_url,
            "filename": filename,
            "timestamp": int(time.time())
        })
        os.remove(filename)
    except Exception as e:
        print(f"❌ Error during video upload: {e}")


# === MAIN ===
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Could not open video/camera: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mid_x, mid_y = w // 2, h // 2

    frame_id, chunk_index = 0, 0
    chunk_start_time = time.time()
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_filename = f"processed_chunk_{chunk_index}.mp4"
    out = cv2.VideoWriter(video_filename, fourcc, fps, (w, h))

    from collections import deque
    zone_history = {f"Zone{i}": deque(maxlen=3) for i in range(1, 5)}
    zone_data = {f"Zone{i}": {"count": 0, "density": "Low"} for i in range(1, 5)}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        vis = frame.copy()
        person_boxes = []

        if frame_id % INFER_EVERY_N_FRAMES == 0:
            results = model(frame, imgsz=640, conf=0.50, verbose=False)[0]
            if results.boxes is not None:
                for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                    if int(cls) == 0:
                        person_boxes.append(list(map(int, box)))

            raw_zone_counts = {f"Zone{i}": 0 for i in range(1, 5)}
            for (x1, y1, x2, y2) in person_boxes:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cx < mid_x and cy < mid_y: raw_zone_counts["Zone1"] += 1
                elif cx >= mid_x and cy < mid_y: raw_zone_counts["Zone2"] += 1
                elif cx < mid_x and cy >= mid_y: raw_zone_counts["Zone3"] += 1
                else: raw_zone_counts["Zone4"] += 1

            for z, c in raw_zone_counts.items():
                zone_history[z].append(c)
                avg_count = sum(zone_history[z]) / len(zone_history[z])
                zone_data[z]["count"] = round(avg_count)
                zone_data[z]["density"] = get_density_label(zone_data[z]["count"])

            if bucket:
                db.reference("/crowd/zones").set(zone_data)

            # ✅ check density for announcements
            maybe_trigger_announcement(zone_data)

        # --- Draw bounding boxes ---
        for (x1, y1, x2, y2) in person_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- Draw zone dividers ---
        cv2.line(vis, (mid_x, 0), (mid_x, h), (255, 255, 255), 2)
        cv2.line(vis, (0, mid_y), (w, mid_y), (255, 255, 255), 2)

        # --- Show zone counts + density ---
        cv2.putText(vis,
                    f"Zone 1: {zone_data['Zone1']['count']} ({zone_data['Zone1']['density']})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if zone_data['Zone1']['density'] != "Low" else (0, 255, 0), 2)

        cv2.putText(vis,
                    f"Zone 2: {zone_data['Zone2']['count']} ({zone_data['Zone2']['density']})",
                    (mid_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if zone_data['Zone2']['density'] != "Low" else (0, 255, 0), 2)

        cv2.putText(vis,
                    f"Zone 3: {zone_data['Zone3']['count']} ({zone_data['Zone3']['density']})",
                    (10, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if zone_data['Zone3']['density'] != "Low" else (0, 255, 0), 2)

        cv2.putText(vis,
                    f"Zone 4: {zone_data['Zone4']['count']} ({zone_data['Zone4']['density']})",
                    (mid_x + 10, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if zone_data['Zone4']['density'] != "Low" else (0, 255, 0), 2)

        out.write(vis)

        # --- Upload frame to Firebase ---
        if bucket and frame_id % UPLOAD_EVERY_N_FRAMES == 0:
            _, buffer = cv2.imencode(".jpg", vis)
            blob = bucket.blob("live/latest_frame.jpg")
            blob.upload_from_string(buffer.tobytes(), content_type="image/jpeg")
            blob.make_public()
            db.reference("/crowd/latest_frame").set({"url": blob.public_url})

        # --- Handle chunk rotation ---
        if time.time() - chunk_start_time >= VIDEO_CHUNK_SECONDS:
            out.release()
            upload_video_chunk(video_filename)
            chunk_index += 1
            video_filename = f"processed_chunk_{chunk_index}.mp4"
            out = cv2.VideoWriter(video_filename, fourcc, fps, (w, h))
            chunk_start_time = time.time()

    out.release()
    upload_video_chunk(video_filename)
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Script finished.")


if __name__ == "__main__":
    main()