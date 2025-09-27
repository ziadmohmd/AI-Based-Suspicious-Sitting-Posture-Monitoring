import os
import cv2
import time
import csv
import math
import threading
import sqlite3
import base64
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, Response, jsonify, request
import numpy as np
import pandas as pd
import mediapipe as mp
import plotly.express as px
from plotly.offline import plot 
from insightface.app import FaceAnalysis

try:
    from plyer import notification
    PLYER_AVAILABLE = True
except Exception:
    PLYER_AVAILABLE = False
    print("Plyer not available; notifications disabled")

# Suppress FutureWarning for insightface and protobuf warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.getLogger('google.protobuf').setLevel(logging.ERROR)

# ------------------ CONFIG ------------------ #
CSV_FILE = "posture_history.csv"
DB_FILE = "employees.db"
SNAPSHOT_DIR = "captured_dashboards"
FACE_DATA_DIR = "static/face_data"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(FACE_DATA_DIR, exist_ok=True)

# Logging cadence (seconds)
LOG_INTERVAL_SEC = 3.0

# Inactivity and away detection
STATE_NOTIFY_THRESHOLD = 5.0
STATE_NOTIFY_COOLDOWN = 60.0
EYE_INACTIVE_THRESHOLD_SEC = 5.0

# Posture alert thresholds
POSTURE_NOTIFY_THRESHOLD = 20.0
POSTURE_NOTIFY_COOLDOWN = 60.0

# Eye detection
EAR_THRESHOLD = 0.28
EYE_STATE_WINDOW = 5

# MJPEG frame size (output)
FRAME_WIDTH = 960

# Time zone for IST (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Face recognition model (ArcFace via InsightFace)
face_model = FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0, det_size=(320, 320))

# ------------------ FLASK & STATE ------------------ #
app = Flask(__name__)

state_lock = threading.Lock()
streaming_active = False

current_posture = "Away"
last_posture = None
posture_start_time = datetime.now(IST)
posture_durations = {k: 0.0 for k in ["Good", "Slouched", "Leaning", "Away", "Inactive"]}

current_eye_state = "Unknown"
last_eye_closed_time = None
last_state_change = None
last_inactive_notification = datetime.min.replace(tzinfo=IST)
last_away_notification = datetime.min.replace(tzinfo=IST)
last_posture_notification = datetime.min.replace(tzinfo=IST)

latest_frame = None
cap = None
worker_thread = None
stop_worker = threading.Event()

last_log_time = datetime.now(IST)
latest_angles = {}
current_employee_id = None
current_employee_name = "Unknown"
eye_state_buffer = []

# ------------------ DATABASE INIT ------------------ #
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# ------------------ FACE MATCHING ------------------ #
def match_embedding(emb):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, embedding FROM employees")
    candidates = c.fetchall()
    conn.close()
    if not candidates:
        return None, "Unknown"
    emb = emb / np.linalg.norm(emb)
    max_sim = -1
    match_id = None
    match_name = "Unknown"
    for eid, name, blob in candidates:
        try:
            cand_emb = np.frombuffer(blob, dtype=np.float32).copy()
            cand_emb = cand_emb / np.linalg.norm(cand_emb)
            sim = np.dot(emb, cand_emb)
            if sim > max_sim:
                max_sim = sim
                match_id = eid
                match_name = name
        except Exception as e:
            print(f"Error processing embedding for {name} (ID: {eid}): {e}")
            continue
    if max_sim > 0.5:
        return match_id, match_name
    return None, "Unknown"

# ------------------ TRAINING ------------------ #
def train_model():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT id, name FROM employees")
        employees = c.fetchall()
        c.execute("DELETE FROM employees")
        conn.commit()

        for emp_id, emp_name in employees:
            emp_dir = os.path.join(FACE_DATA_DIR, str(emp_id))
            if not os.path.exists(emp_dir):
                print(f"Directory {emp_dir} not found, skipping {emp_name}")
                continue
            images = [os.path.join(emp_dir, f) for f in os.listdir(emp_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                print(f"No images found for {emp_name} (ID: {emp_id})")
                continue
            img_path = images[0]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to load image {img_path}")
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_model.get(rgb)
            if not faces:
                print(f"No face detected in {img_path}")
                continue
            embedding = faces[0].embedding.tobytes()
            c.execute("INSERT INTO employees (id, name, embedding) VALUES (?, ?, ?)", 
                     (emp_id, emp_name, embedding))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error training model: {e}")
        return False

# ------------------ MODELS ------------------ #
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ------------------ UTILITIES ------------------ #
def angle_3pts(a, b, c):
    try:
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        ang = math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))
        return ang
    except Exception:
        return np.nan

def vertical_angle(p1, p2):
    try:
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        vertical = np.array([0, -1.0])
        cosang = np.dot(v, vertical) / (np.linalg.norm(v) * np.linalg.norm(vertical) + 1e-9)
        ang = math.degrees(math.acos(np.clip(cosang, -1.0, 1.0)))
        return ang
    except Exception:
        return np.nan

def compute_ear(eye_points, landmarks, w, h):
    try:
        points = [(landmarks[idx].x * w, landmarks[idx].y * h) for idx in eye_points]
        A = math.hypot(points[1][0] - points[5][0], points[1][1] - points[5][1])
        B = math.hypot(points[2][0] - points[4][0], points[2][1] - points[4][1])
        C = math.hypot(points[0][0] - points[3][0], points[0][1] - points[3][1])
        ear = (A + B) / (2.0 * C + 1e-9)
        return ear
    except Exception:
        return np.nan

# ------------------ CLASSIFIER ------------------ #
def classify_posture_and_eyes(landmarks, face_landmarks, w, h):
    global eye_state_buffer
    if landmarks is None:
        return "Away", {}, "Unknown"

    def get_xy(idx, lm_list=landmarks):
        lm = lm_list[idx]
        return (lm.x * w, lm.y * h)

    try:
        NOSE = mp_pose.PoseLandmark.NOSE.value
        L_SH = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        R_SH = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
        L_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
        R_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
        L_ELB = mp_pose.PoseLandmark.LEFT_ELBOW.value
        R_ELB = mp_pose.PoseLandmark.RIGHT_ELBOW.value
        L_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value
        R_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value
        L_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
        R_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value

        nose = get_xy(NOSE)
        lsh = get_xy(L_SH); rsh = get_xy(R_SH)
        lhip = get_xy(L_HIP); rhip = get_xy(R_HIP)
        lelb = get_xy(L_ELB); relb = get_xy(R_ELB)
        lwri = get_xy(L_WRIST); rwri = get_xy(R_WRIST)
        lknee = get_xy(L_KNEE); rknee = get_xy(R_KNEE)

        shoulder_mid = ((lsh[0] + rsh[0]) / 2, (lsh[1] + rsh[1]) / 2)
        hip_mid = ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)

        neck_flex = vertical_angle(shoulder_mid, nose)
        torso_tilt = vertical_angle(hip_mid, shoulder_mid)
        left_elbow = angle_3pts(lsh, lelb, lwri)
        right_elbow = angle_3pts(rsh, relb, rwri)
        left_hip_angle = angle_3pts(lsh, lhip, lknee)
        right_hip_angle = angle_3pts(rsh, rhip, rknee)
        shoulder_delta_y = abs(lsh[1] - rsh[1]) / max(h, 1)

        angles = {
            "neck_flex_deg": round(neck_flex, 1) if not np.isnan(neck_flex) else np.nan,
            "torso_tilt_deg": round(torso_tilt, 1) if not np.isnan(torso_tilt) else np.nan,
            "left_elbow_deg": round(left_elbow, 1) if not np.isnan(left_elbow) else np.nan,
            "right_elbow_deg": round(right_elbow, 1) if not np.isnan(right_elbow) else np.nan,
            "left_hip_deg": round(left_hip_angle, 1) if not np.isnan(left_hip_angle) else np.nan,
            "right_hip_deg": round(right_hip_angle, 1) if not np.isnan(right_hip_angle) else np.nan,
            "shoulder_delta_y": round(shoulder_delta_y, 3),
        }

        SLOUCH_NECK_DEG = 25.0
        SLOUCH_TORSO_DEG = 20.0
        LEAN_DELTA_Y = 0.05

        posture = "Good"
        if shoulder_delta_y > LEAN_DELTA_Y:
            posture = "Leaning"
        if neck_flex > SLOUCH_NECK_DEG or torso_tilt > SLOUCH_TORSO_DEG:
            posture = "Slouched"

        eye_state = "Unknown"
        if face_landmarks and face_landmarks[0]:
            left_eye_points = [362, 385, 387, 263, 373, 380]
            right_eye_points = [33, 160, 158, 133, 153, 144]
            left_ear = compute_ear(left_eye_points, face_landmarks[0].landmark, w, h)
            right_ear = compute_ear(right_eye_points, face_landmarks[0].landmark, w, h)

            if not np.isnan(left_ear) and not np.isnan(right_ear):
                avg_ear = (left_ear + right_ear) / 2
                current_eye_state = "Open" if avg_ear > EAR_THRESHOLD else "Closed"
            else:
                current_eye_state = "Unknown"

            eye_state_buffer.append(current_eye_state)
            if len(eye_state_buffer) > EYE_STATE_WINDOW:
                eye_state_buffer.pop(0)
            counts = {}
            for state in eye_state_buffer:
                counts[state] = counts.get(state, 0) + 1
            eye_state = max(counts, key=counts.get, default="Unknown")
        else:
            eye_state_buffer.append("Unknown")
            if len(eye_state_buffer) > EYE_STATE_WINDOW:
                eye_state_buffer.pop(0)
            eye_state = "Unknown"

        return posture, angles, eye_state
    except Exception as e:
        print(f"Error in classify_posture_and_eyes: {e}")
        return "Away", {}, "Unknown"

# ------------------ LOGGING ------------------ #
def ensure_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "employee_id", "employee_name", "posture", "eye_state",
                "neck_flex_deg", "torso_tilt_deg",
                "left_elbow_deg", "right_elbow_deg",
                "left_hip_deg", "right_hip_deg",
                "shoulder_delta_y"
            ])

def log_row(posture, angles, eye_state, employee_id=None, employee_name="Unknown"):
    global last_log_time
    now = datetime.now(IST)
    if (now - last_log_time).total_seconds() < LOG_INTERVAL_SEC:
        return
    last_log_time = now

    ensure_csv()
    row = [
        now.strftime("%Y-%m-%d %H:%M:%S"),
        employee_id if employee_id is not None else "",
        employee_name,
        posture,
        eye_state,
        angles.get("neck_flex_deg", ""),
        angles.get("torso_tilt_deg", ""),
        angles.get("left_elbow_deg", ""),
        angles.get("right_elbow_deg", ""),
        angles.get("left_hip_deg", ""),
        angles.get("right_hip_deg", ""),
        angles.get("shoulder_delta_y", ""),
    ]
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception as e:
        print(f"Error logging to CSV: {e}")

# ------------------ NOTIFICATIONS ------------------ #
def maybe_notify_state(posture, eye_state, employee_name="Unknown"):
    global last_state_change, last_inactive_notification, last_away_notification, last_posture_notification
    now = datetime.now(IST)

    if posture in ["Inactive", "Away"] or eye_state == "Closed":
        if last_state_change is None:
            last_state_change = now
        state_duration = (now - last_state_change).total_seconds()

        if state_duration >= STATE_NOTIFY_THRESHOLD:
            if posture == "Inactive" or eye_state == "Closed":
                since_last_notif = (now - last_inactive_notification).total_seconds()
                if since_last_notif >= STATE_NOTIFY_COOLDOWN and PLYER_AVAILABLE:
                    try:
                        message = (
                            f"{employee_name}, eyes closed for too long, indicating inactivity." if eye_state == "Closed"
                            else f"{employee_name}, you've been inactive for a while. Time to move!"
                        )
                        notification.notify(
                            title="Posture Monitor",
                            message=message,
                            app_name="Posture Monitor",
                            timeout=10
                        )
                        print(f"Inactive notification sent: {message}")
                        last_inactive_notification = now
                    except Exception as e:
                        print(f"Error sending inactive notification: {e}")
            elif posture == "Away":
                since_last_notif = (now - last_away_notification).total_seconds()
                if since_last_notif >= STATE_NOTIFY_COOLDOWN and PLYER_AVAILABLE:
                    try:
                        notification.notify(
                            title="Posture Monitor",
                            message=f"{employee_name}, you've been away for a while. Please return to your workstation.",
                            app_name="Posture Monitor",
                            timeout=10
                        )
                        print(f"Away notification sent: {employee_name}")
                        last_away_notification = now
                    except Exception as e:
                        print(f"Error sending away notification: {e}")
    elif posture in ["Slouched", "Leaning"]:
        if last_state_change is None:
            last_state_change = now
        state_duration = (now - last_state_change).total_seconds()
        since_last_posture_notif = (now - last_posture_notification).total_seconds()

        if state_duration >= POSTURE_NOTIFY_THRESHOLD and since_last_posture_notif >= POSTURE_NOTIFY_COOLDOWN and PLYER_AVAILABLE:
            try:
                notification.notify(
                    title="Posture Monitor",
                    message=f"{employee_name}, you have been {posture.lower()} for over {int(POSTURE_NOTIFY_THRESHOLD)} seconds. Please correct your posture.",
                    app_name="Posture Monitor",
                    timeout=10
                )
                print(f"Posture notification sent: {employee_name} is {posture.lower()}")
                last_posture_notification = now
            except Exception as e:
                print(f"Error sending posture notification: {e}")
    else:
        last_state_change = None

# ------------------ DASHBOARD ------------------ #
def build_dashboard_fig(date_start=None, date_end=None, sort_by='timestamp', employee_id=None):
    ensure_csv()
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"CSV loaded successfully, rows: {len(df)}")
    except Exception as e:
        print(f"Error reading CSV for dashboard: {e}")
        df = pd.DataFrame(columns=["timestamp", "employee_id", "employee_name", "posture", "eye_state"])

    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if employee_id is not None:
            df = df[df["employee_id"] == int(employee_id)]
        if date_start:
            date_start = pd.to_datetime(date_start)
            df = df[df["timestamp"] >= date_start]
        if date_end:
            date_end = pd.to_datetime(date_end)
            df = df[df["timestamp"] <= date_end]
        if sort_by in df.columns:
            df = df.sort_values(by=sort_by)
        else:
            df = df.sort_values(by="timestamp")

    if df.empty:
        print("CSV is empty or filtered to empty, generating placeholder charts")
        empty_posture = pd.DataFrame({"posture": ["Good", "Slouched", "Leaning", "Away", "Inactive"], "count": [0,0,0,0,0]})
        empty_eyes = pd.DataFrame({"eye_state": ["Open","Closed","Unknown"], "count":[0,0,0]})
        empty_angles = pd.DataFrame({"angle": ["Neck Flex", "Torso Tilt", "Left Elbow", "Right Elbow", "Left Hip", "Right Hip"], "average": [0,0,0,0,0,0]})

        fig1 = px.bar(empty_posture, x="posture", y="count", title="Posture Frequency (No data yet)")
        fig2 = px.bar(empty_eyes, x="eye_state", y="count", title="Eye State Frequency (No data yet)")
        fig3 = px.pie(empty_posture, values="count", names="posture", title="Posture Distribution (No data yet)")
        fig4 = px.pie(empty_eyes, values="count", names="eye_state", title="Eye State Distribution (No data yet)")
        fig5 = px.line(title="Posture Over Time (No data yet)")
        fig6 = px.bar(empty_angles, x="angle", y="average", title="Average Body Angles (No data yet)")
        fig7 = px.pie(empty_eyes, values="count", names="eye_state", title="Eye State by Hour (No data yet)")
        return fig1, fig2, fig3, fig4, fig5, fig6, fig7

    counts = df["posture"].value_counts().reset_index()
    counts.columns = ["posture", "count"]
    fig1 = px.bar(counts, x="posture", y="count", title="Posture Frequency", text="count")
    fig1.update_traces(textposition='outside')

    if "eye_state" in df.columns:
        eye_counts = df["eye_state"].value_counts().reset_index()
        eye_counts.columns = ["eye_state", "count"]
        fig2 = px.bar(eye_counts, x="eye_state", y="count", title="Eye State Frequency", text="count")
        fig2.update_traces(textposition='outside')
    else:
        empty_eyes = pd.DataFrame({"eye_state": ["Open","Closed","Unknown"], "count":[0,0,0]})
        fig2 = px.bar(empty_eyes, x="eye_state", y="count", title="Eye State Frequency (No data yet)")

    fig3 = px.pie(counts, values="count", names="posture", title="Posture Distribution")

    if "eye_state" in df.columns and not df["eye_state"].empty:
        eye_counts = df["eye_state"].value_counts().reset_index()
        eye_counts.columns = ["eye_state", "count"]
        fig4 = px.pie(eye_counts, values="count", names="eye_state", title="Eye State Distribution")
    else:
        empty_eyes = pd.DataFrame({"eye_state": ["Open","Closed","Unknown"], "count":[0,0,0]})
        fig4 = px.pie(empty_eyes, values="count", names="eye_state", title="Eye State Distribution (No data yet)")

    if not df.empty and "timestamp" in df.columns:
        df["posture_numeric"] = df["posture"].map({"Good": 1, "Slouched": 2, "Leaning": 3, "Away": 4, "Inactive": 5})
        fig5 = px.line(df, x="timestamp", y="posture_numeric", title="Posture Over Time",
                       labels={"posture_numeric": "Posture Level"}, markers=True)
        fig5.update_yaxes(tickvals=[1,2,3,4,5], ticktext=["Good", "Slouched", "Leaning", "Away", "Inactive"])
    else:
        fig5 = px.line(title="Posture Over Time (No data yet)")

    angle_cols = ["neck_flex_deg", "torso_tilt_deg", "left_elbow_deg", "right_elbow_deg", "left_hip_deg", "right_hip_deg"]
    angle_labels = ["Neck Flex", "Torso Tilt", "Left Elbow", "Right Elbow", "Left Hip", "Right Hip"]
    averages = []
    for col in angle_cols:
        if col in df.columns:
            avg = df[col].dropna().mean()
            averages.append(avg if not np.isnan(avg) else 0)
        else:
            averages.append(0)
    angles_df = pd.DataFrame({"angle": angle_labels, "average": averages})
    fig6 = px.bar(angles_df, x="angle", y="average", title="Average Body Angles (Degrees)", text="average")
    fig6.update_traces(textposition='outside')

    if "eye_state" in df.columns and "timestamp" in df.columns and not df.empty:
        df["hour"] = df["timestamp"].dt.hour
        eye_by_hour = df.groupby("hour")["eye_state"].value_counts().unstack().fillna(0)
        fig7 = px.pie(eye_counts, values="count", names="eye_state", title="Eye State Distribution")
    else:
        empty_eyes = pd.DataFrame({"eye_state": ["Open","Closed","Unknown"], "count":[0,0,0]})
        fig7 = px.pie(empty_eyes, values="count", names="eye_state", title="Eye State by Hour (No data yet)")

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

# ------------------ CAMERA UTILITIES ------------------ #
def try_open_camera(index, backend=cv2.CAP_DSHOW):
    try:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"Camera opened successfully on index {index} with backend {backend}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
        else:
            print(f"Failed to open camera on index {index} with backend {backend}")
            cap.release()
            return None
    except Exception as e:
        print(f"Error opening camera on index {index} with backend {backend}: {e}")
        return None

# ------------------ WORKER ------------------ #
def camera_worker():
    global latest_frame, current_posture, last_posture, posture_start_time
    global posture_durations, streaming_active, cap, latest_angles
    global current_eye_state, last_eye_closed_time, last_state_change
    global current_employee_id, current_employee_name, eye_state_buffer

    while not stop_worker.is_set():
        if not streaming_active:
            time.sleep(0.05)
            continue

        # Try to open camera with multiple indices and backends
        if cap is None or not cap.isOpened():
            print("Attempting to initialize camera...")
            for index in range(4):  # Try indices 0 to 3
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                    cap = try_open_camera(index, backend)
                    if cap is not None:
                        break
                if cap is not None:
                    break
            if cap is None:
                print("All camera indices and backends failed. Retrying in 5 seconds...")
                time.sleep(5)
                continue

        max_retries = 3
        ok, frame = False, None
        for attempt in range(max_retries):
            try:
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                print(f"Frame capture failed (attempt {attempt + 1}/{max_retries})")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error reading frame (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(0.1)
        if not ok or frame is None:
            print("Frame capture failed after retries. Reinitializing camera...")
            cap.release()
            cap = None
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb)
        face_results = face_mesh.process(rgb)

        # Face recognition
        employee_id = None
        employee_name = "Unknown"
        try:
            faces = face_model.get(rgb)
            if faces:
                emb = faces[0].embedding
                employee_id, employee_name = match_embedding(emb)
        except Exception as e:
            print(f"Error in face recognition: {e}")
            employee_id, employee_name = None, "Unknown"

        # Posture and eye classification
        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            posture, angles, eye_state = classify_posture_and_eyes(
                pose_results.pose_landmarks.landmark,
                face_results.multi_face_landmarks,
                w, h
            )
            if eye_state == "Closed":
                if last_eye_closed_time is None:
                    last_eye_closed_time = datetime.now(IST)
                eye_inactive_duration = (datetime.now(IST) - last_eye_closed_time).total_seconds()
                if eye_inactive_duration > EYE_INACTIVE_THRESHOLD_SEC:
                    posture = "Inactive"
            else:
                last_eye_closed_time = None
        else:
            posture, angles, eye_state = "Away", {}, "Unknown"
            eye_state_buffer.append("Unknown")
            if len(eye_state_buffer) > EYE_STATE_WINDOW:
                eye_state_buffer.pop(0)

        # Update state
        now = datetime.now(IST)
        with state_lock:
            if last_posture is None:
                last_posture = posture
                posture_start_time = now
                last_state_change = now
            if posture != last_posture:
                elapsed = (now - posture_start_time).total_seconds()
                posture_durations[last_posture] += elapsed
                last_posture = posture
                posture_start_time = now
                last_state_change = now

            current_posture = posture
            current_eye_state = eye_state
            latest_angles = angles
            current_employee_id = employee_id
            current_employee_name = employee_name

            # Update latest_frame safely
            try:
                scale = FRAME_WIDTH / float(w)
                resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                latest_frame = resized_frame.copy()
            except Exception as e:
                print(f"Error resizing frame: {e}")
                latest_frame = None

        # Log and notify
        log_row(posture, angles, eye_state, employee_id, employee_name)
        print(f"Logged posture={posture}, eye={eye_state}, employee={employee_name} (ID: {employee_id})")
        maybe_notify_state(posture, eye_state, employee_name)

        # Overlay text
        color = (0, 200, 0) if posture == "Good" else (0, 165, 255) if posture in ["Leaning", "Away"] else (0, 0, 255)
        cv2.putText(frame, f"Posture: {posture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Eyes: {eye_state}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Employee: {employee_name} (ID: {employee_id if employee_id else 'N/A'})", 
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# ------------------ ROUTES ------------------ #
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/register_employee", methods=["POST"])
def register_employee():
    data = request.json
    name = data.get('name')
    image_base64 = data.get('image')
    if not name or not image_base64:
        return jsonify({"error": "Missing name or image"}), 400
    try:
        img_data = base64.b64decode(image_base64.split(',')[1])
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_model.get(rgb)
        if not faces:
            return jsonify({"error": "No face detected"}), 400
        embedding = faces[0].embedding.tobytes()
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO employees (name, embedding) VALUES (?, ?)", (name, embedding))
        employee_id = c.lastrowid
        conn.commit()
        conn.close()
        emp_dir = os.path.join(FACE_DATA_DIR, str(employee_id))
        os.makedirs(emp_dir, exist_ok=True)
        img_path = os.path.join(emp_dir, f"{employee_id}_1.png")
        cv2.imwrite(img_path, img)
        return jsonify({"ok": True, "employee_id": employee_id})
    except Exception as e:
        print(f"Error registering employee: {e}")
        return jsonify({"error": "Registration failed"}), 500

@app.route("/train_model", methods=["POST"])
def train_model_route():
    if train_model():
        return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Training failed"})

@app.route("/employees", methods=["GET"])
def employees():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name FROM employees")
    emps = c.fetchall()
    conn.close()
    return jsonify(emps)

@app.route("/start", methods=["POST"])
def start():
    global streaming_active, last_posture, posture_start_time, posture_durations, last_state_change
    with state_lock:
        streaming_active = True
        posture_durations = {k: 0.0 for k in ["Good", "Slouched", "Leaning", "Away", "Inactive"]}
        posture_start_time = datetime.now(IST)
        last_posture = None
        last_state_change = None
    print("Streaming started")
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    global streaming_active, last_posture, posture_start_time, posture_durations
    with state_lock:
        now = datetime.now(IST)
        if last_posture is not None:
            posture_durations[last_posture] += (now - posture_start_time).total_seconds()
            posture_start_time = now
        streaming_active = False
    print("Streaming stopped")
    return jsonify({"ok": True})

@app.route("/video_feed")
def video_feed():
    def gen():
        frame_count = 0
        while True:
            with state_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                print(f"Frame {frame_count}: No frame available for streaming")
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera unavailable", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                success, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            else:
                try:
                    success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if not success:
                        print(f"Frame {frame_count}: Failed to encode frame")
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(blank, "Encoding error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        success, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                except Exception as e:
                    print(f"Frame {frame_count}: Error encoding frame: {e}")
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Encoding error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    success, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            
            if success:
                jpg = buffer.tobytes()
                print(f"Frame {frame_count}: Sending frame, size {len(jpg)} bytes")
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Cache-Control: no-cache, no-store, must-revalidate\r\n'
                       b'Pragma: no-cache\r\n'
                       b'Expires: 0\r\n\r\n' + jpg + b'\r\n')
            else:
                print(f"Frame {frame_count}: Failed to prepare frame for streaming")
                time.sleep(0.1)
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS

    print("Starting video feed stream")
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/metrics")
def metrics():
    with state_lock:
        now = datetime.now(IST)
        durations = posture_durations.copy()
        if streaming_active and last_posture is not None:
            durations[last_posture] = durations.get(last_posture, 0.0) + (now - posture_start_time).total_seconds()
        posture_now = current_posture
        eye_state_now = current_eye_state
        angles = latest_angles.copy()
        emp_id = current_employee_id
        emp_name = current_employee_name

    return jsonify({
        "posture": posture_now,
        "eye_state": eye_state_now,
        "employee_name": emp_name,
        "durations_sec": durations,
        "angles": angles,
        "employee_id": emp_id
    })

@app.route("/dashboard_html")
def dashboard_html():
    try:
        date_start = request.args.get('date_start')
        date_end = request.args.get('date_end')
        sort_by = request.args.get('sort_by', 'timestamp')
        employee_id = request.args.get('employee_id')
        fig1, fig2, fig3, fig4, fig5, fig6, fig7 = build_dashboard_fig(date_start, date_end, sort_by, employee_id)
        html = (
            '<html><head><title>Dashboard</title></head><body style="background:#0d142a;color:#eef1f7;">'
            + plot(fig1, output_type="div", include_plotlyjs="cdn")
            + plot(fig2, output_type="div", include_plotlyjs=False)
            + plot(fig3, output_type="div", include_plotlyjs=False)
            + plot(fig4, output_type="div", include_plotlyjs=False)
            + plot(fig5, output_type="div", include_plotlyjs=False)
            + plot(fig6, output_type="div", include_plotlyjs=False)
            + plot(fig7, output_type="div", include_plotlyjs=False)
            + '</body></html>'
        )
        return html
    except Exception as e:
        print(f"Error generating dashboard HTML: {e}")
        return (
            '<html><head><title>Dashboard Error</title></head>'
            '<body style="background:#0d142a;color:#eef1f7;text-align:center;padding:20px;">'
            '<p>Error generating dashboard. Check server logs or ensure posture_history.csv exists.</p>'
            '</body></html>'
        )

def save_dashboard_html(employee_id=None):
    try:
        fig1, fig2, fig3, fig4, fig5, fig6, fig7 = build_dashboard_fig(employee_id=employee_id)
        html = (
            '<html><head><title>Dashboard Snapshot</title></head><body style="background:#0d142a;color:#eef1f7;">'
            + plot(fig1, output_type="div", include_plotlyjs="cdn")
            + plot(fig2, output_type="div", include_plotlyjs=False)
            + plot(fig3, output_type="div", include_plotlyjs=False)
            + plot(fig4, output_type="div", include_plotlyjs=False)
            + plot(fig5, output_type="div", include_plotlyjs=False)
            + plot(fig6, output_type="div", include_plotlyjs=False)
            + plot(fig7, output_type="div", include_plotlyjs=False)
            + '</body></html>'
        )
        now = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_{now}{'_emp' + str(employee_id) if employee_id else ''}.html"
        path = os.path.join(SNAPSHOT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path
    except Exception as e:
        print(f"Error saving dashboard: {e}")
        return None

@app.route("/save_dashboard", methods=["POST"])
def save_dashboard():
    employee_id = request.json.get('employee_id')
    path = save_dashboard_html(employee_id)
    if path:
        return jsonify({"ok": True, "path": path})
    return jsonify({"ok": False, "error": "Failed to save dashboard"})

@app.route("/debug_csv", methods=["GET"])
def debug_csv():
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            return jsonify({"rows": len(df), "columns": list(df.columns), "sample": df.head().to_dict(orient="records")})
        return jsonify({"error": "CSV file not found"})
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"})

# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    init_db()
    worker_thread = threading.Thread(target=camera_worker, daemon=True)
    worker_thread.start()
    ensure_csv()
    print("Open http://127.0.0.1:5000")
    print("Admin Panel at http://127.0.0.1:5000/admin")
    print("Debug CSV at http://127.0.0.1:5000/debug_csv")
    app.run(host="127.0.0.1", port=5000, debug=False)