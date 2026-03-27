from ultralytics import YOLO
import cv2
import mediapipe as mp
import urllib.request
import os

# =========================
# DOWNLOAD MEDIAPIPE MODEL
# =========================
model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

# =========================
# LOAD MODELS
# =========================
letter_model = YOLO("best (4).pt")
word_model = YOLO("best_word.pt")

# =========================
# MEDIAPIPE TASKS API
# =========================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)  # 🔥 reduce lag

timestamp_ms = 0
frame_count = 0  # 🔥 for skipping frames

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        h, w, _ = frame.shape

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                # Bounding box
                x_coords = [lm.x * w for lm in hand_landmarks]
                y_coords = [lm.y * h for lm in hand_landmarks]

                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                # Padding
                pad = 80
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(w, x_max + pad)
                y_max = min(h, y_max + pad)

                hand_crop = frame[y_min:y_max, x_min:x_max]

                # =========================
                # OPTIMIZED INFERENCE
                # =========================
                if hand_crop.size != 0 and frame_count % 3 == 0:

                    # 🔥 smaller image = faster
                    hand_crop = cv2.resize(hand_crop, (320, 320))

                    best_word = None
                    best_conf = 0

                    # 🔥 run word model less frequently
                    if frame_count % 2 == 0:
                        word_results = word_model(hand_crop, conf=0.5)

                        for r in word_results:
                            for box in r.boxes:
                                conf = float(box.conf[0])
                                label = word_model.names[int(box.cls[0])]

                                if conf > best_conf:
                                    best_conf = conf
                                    best_word = label

                    # =========================
                    # WORD DETECTION
                    # =========================
                    if best_word and best_conf < 0.6:
                        cv2.putText(frame, best_word,
                                    (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    2,
                                    (255, 0, 0),
                                    4)

                    else:
                        # =========================
                        # LETTER DETECTION
                        # =========================
                        letter_results = letter_model(hand_crop, conf=0.5)

                        best_letter = None
                        best_conf_l = 0

                        for r in letter_results:
                            for box in r.boxes:
                                conf = float(box.conf[0])
                                label = letter_model.names[int(box.cls[0])]

                                if conf > best_conf_l:
                                    best_conf_l = conf
                                    best_letter = label

                        if best_letter:
                            cv2.putText(frame, best_letter,
                                        (50, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        2,
                                        (0, 255, 0),
                                        4)

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

        # =========================
        # DISPLAY
        # =========================
        cv2.imshow("ASL Detection (Optimized)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()