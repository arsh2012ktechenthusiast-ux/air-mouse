import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

# Initialize screen resolution and disable PyAutoGUI fail-safe
screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

# Configure MediaPipe HandLandmarker Options for VIDEO mode
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize Webcam Stream and reduce resolution to 320x240 for maximum speed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Smoothing & Click Debounce Variables
prev_x, prev_y = 0, 0
smoothing_factor = 0.5  
last_click_time = 0
click_cooldown = 0.3  

start_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally for natural mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert OpenCV BGR frame to MediaPipe Image format (RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )

    # Generate continuous timestamp in milliseconds for VIDEO mode
    timestamp_ms = int((time.time() - start_time) * 1000)

    # Run Hand Landmark Detection
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    # Process detected hand landmarks and draw visual dots
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw circles on all landmarks for visual feedback
            for landmark in hand_landmarks:
                lx, ly = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (lx, ly), 4, (0, 255, 0), -1)

            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            # Map normalized coordinates to pixel screen space
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)

            # Translate to absolute display coordinates
            target_x = int(screen_w * (index_x / w))
            target_y = int(screen_h * (index_y / h))

            # Apply Exponential Moving Average (EMA) for smooth movement
            screen_x = int(prev_x + (target_x - prev_x) * smoothing_factor)
            screen_y = int(prev_y + (target_y - prev_y) * smoothing_factor)
            
            pyautogui.moveTo(screen_x, screen_y)
            prev_x, prev_y = screen_x, screen_y

            # Distance check and click debounce
            distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
            current_time = time.time()
            if distance < 30 and (current_time - last_click_time) > click_cooldown:
                pyautogui.click()
                last_click_time = current_time

    # Render feedback window
    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()