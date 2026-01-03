import cv2
import time
import torch
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# Disable gradients for inference
torch.set_grad_enabled(False)

model_path = "models/tusimple_18.pth"
model_type = ModelType.TUSIMPLE
use_gpu = False

lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

cap = cv2.VideoCapture("Input/Input2.mp4")  # or 0 for webcam
cv2.namedWindow("Detected Lanes", cv2.WINDOW_NORMAL)

prev_time = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Skip frames (important for speed)
    if frame_count % 3 != 0:
        continue

    # Resize frame
    frame = cv2.resize(frame, (512, 288))
    h, w, _ = frame.shape

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Lane detection
    output_img = lane_detector.detect_lanes(frame)

    # Lane status
    lane_status = "ACTIVE" if output_img is not None else "NOT DETECTED"
    status_color = (0, 255, 0) if lane_status == "ACTIVE" else (0, 0, 255)

    # Overlay text
    cv2.putText(output_img, f"Lane Status: {lane_status}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.putText(output_img, f"FPS: {fps:.2f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(output_img, "Model: UltraFast Lane Detector (TuSimple)",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(output_img, f"Mode: {'GPU' if use_gpu else 'CPU'}",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(output_img, f"Resolution: {w}x{h}",
                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Detected Lanes", output_img)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
