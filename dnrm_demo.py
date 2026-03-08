import cv2
import os
import os
# Mute the "ModelDependencyMissing" warnings
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"
# Mute the "FutureWarning"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import math
from inference import get_model
from ultralytics import YOLO
from dotenv import load_dotenv  # <-- New secure import

# --- 1. SECURITY & ENVIRONMENT SETTINGS ---
load_dotenv()  # This pulls the key from your hidden .env file
api_key = os.getenv("ROBOFLOW_API_KEY")

os.environ["ORT_DEFAULT_LOG_SEVERITY_LEVEL"] = "4"
os.environ["ORT_LOGGING_LEVEL"] = "4"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CoreMLExecutionProvider]"
os.environ["MODEL_CACHE_DIR"] = "./roboflow_model_cache"
os.environ["QWEN_2_5_ENABLED"] = "False"
os.environ["QWEN_3_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"

# --- 2. INITIALIZE MODELS ---
print("Loading Models Securely... This may take a moment.")

if not api_key:
    print("Error: ROBOFLOW_API_KEY not found in .env file!")
    exit()

# The key is hidden inside the 'api_key' variable
roboflow_model = get_model(model_id="dnrm_blur/3", api_key=api_key)

local_face_model = "models/yolov8n-face.pt"
if not os.path.exists(local_face_model):
    print(f"Error: Cannot find {local_face_model}. Make sure the file is in the 'models' folder!")
    exit()

face_model = YOLO(local_face_model)

# --- 3. VIDEO SETUP ---
video_path = "test_video.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('your_video_name.mp4', fourcc, fps, (frame_width, frame_height))
out.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

print("Processing Secure Multi-Target Tracking... Press 'q' to stop.")

# --- 4. DUAL-MEMORY SYSTEM ---
active_trackers = [] 
MAX_MARKER_MISSING = 90  
MAX_FACE_MISSING = 10    

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    valid_markers = []
    detected_faces = []

    # --- A. FIND ALL MARKERS ---
    try:
        results = roboflow_model.infer(frame)[0]
        for pred in results.predictions:
            if pred.confidence >= 0.4:
                valid_markers.append((int(pred.x), int(pred.y)))
                mx, my = int(pred.x - pred.width/2), int(pred.y - pred.height/2)
                cv2.rectangle(frame, (mx, my), (mx + int(pred.width), my + int(pred.height)), (255, 0, 255), 2)
    except Exception:
        pass 

    # --- B. FIND ALL FACES ---
    face_results = face_model.predict(frame, device="mps", verbose=False)
    for box in face_results[0].boxes:
        conf = box.conf[0].item()
        if conf > 0.3:  
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            detected_faces.append({
                'x': x1, 'y': y1, 'w': w, 'h': h,
                'cx': x1 + (w / 2), 'cy': y1 + (h / 2)
            })

    # --- C. THE DUAL-MEMORY ENGINE ---
    current_frame_targets = []

    for mx, my in valid_markers:
        best_distance = float('inf')
        best_face = None
        for face in detected_faces:
            if abs(my - face['cy']) > face['h'] * 2.5: continue
            
            x_dist = (mx - face['cx']) * 3.0 
            y_dist = abs(my - face['cy'])
            distance = math.hypot(x_dist, y_dist)
            
            if distance < best_distance:
                best_distance = distance
                best_face = face.copy()
        
        if best_face:
            is_duplicate = any(f['cx'] == best_face['cx'] and f['cy'] == best_face['cy'] for f in current_frame_targets)
            if not is_duplicate:
                current_frame_targets.append(best_face)

    new_trackers = []
    for target in current_frame_targets:
        matched_tracker = None
        best_tracker_dist = float('inf')
        
        for tracker in active_trackers:
            dist = math.hypot(tracker['face']['cx'] - target['cx'], tracker['face']['cy'] - target['cy'])
            if dist < target['w'] * 1.5 and dist < best_tracker_dist:
                best_tracker_dist = dist
                matched_tracker = tracker
                
        if matched_tracker:
            old_face = matched_tracker['face']
            target['x'] = (target['x'] * 0.6) + (old_face['x'] * 0.4)
            target['y'] = (target['y'] * 0.6) + (old_face['y'] * 0.4)
            target['w'] = (target['w'] * 0.6) + (old_face['w'] * 0.4)
            target['h'] = (target['h'] * 0.6) + (old_face['h'] * 0.4)
            
        new_trackers.append({
            'face': target, 
            'marker_missing': 0, 
            'face_missing': 0
        })

    for tracker in active_trackers:
        is_updated = any(math.hypot(tracker['face']['cx'] - nt['face']['cx'], tracker['face']['cy'] - nt['face']['cy']) < tracker['face']['w'] for nt in new_trackers)
        
        if not is_updated:
            marker_missing = tracker.get('marker_missing', 0) + 1
            if marker_missing > MAX_MARKER_MISSING: continue 
                
            best_distance = float('inf')
            best_face = None
            for face in detected_faces:
                dist = math.hypot(tracker['face']['cx'] - face['cx'], tracker['face']['cy'] - face['cy'])
                if dist < tracker['face']['w'] * 1.5 and dist < best_distance:
                    best_distance = dist
                    best_face = face.copy()
            
            if best_face:
                old_face = tracker['face']
                best_face['x'] = (best_face['x'] * 0.6) + (old_face['x'] * 0.4)
                best_face['y'] = (best_face['y'] * 0.6) + (old_face['y'] * 0.4)
                best_face['w'] = (best_face['w'] * 0.6) + (old_face['w'] * 0.4)
                best_face['h'] = (best_face['h'] * 0.6) + (old_face['h'] * 0.4)
                
                new_trackers.append({
                    'face': best_face, 
                    'marker_missing': marker_missing, 
                    'face_missing': 0 
                })
            else:
                face_missing = tracker.get('face_missing', 0) + 1
                if face_missing <= MAX_FACE_MISSING:
                    new_trackers.append({
                        'face': tracker['face'], 
                        'marker_missing': marker_missing, 
                        'face_missing': face_missing
                    })

    active_trackers = new_trackers

    # --- D. APPLY BLUR TO EVERYONE ---
    for tracker in active_trackers:
        face = tracker['face']
        fx, fy, fw, fh = int(face['x']), int(face['y']), int(face['w']), int(face['h'])
        
        pad_w, pad_h = int(fw * 0.2), int(fh * 0.2)
        bx1, by1 = max(0, fx - pad_w), max(0, fy - pad_h)
        bx2, by2 = min(frame_width, fx + fw + pad_w), min(frame_height, fy + fh + pad_h)

        kernel_size = max(15, int(fw * 0.8))
        if kernel_size % 2 == 0: kernel_size += 1

        face_roi = frame[by1:by2, bx1:bx2]
        if face_roi.size > 0:
            frame[by1:by2, bx1:bx2] = cv2.GaussianBlur(face_roi, (kernel_size, kernel_size), 0)

    out.write(frame)
    cv2.imshow("DNRM Multi-Target Pipeline", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Export complete!")