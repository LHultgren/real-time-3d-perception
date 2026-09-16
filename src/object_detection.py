import cv2
import time
from collections import deque
from ultralytics import YOLO
import torch

def run(source=1, smoothing_window=30):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO("yolov8n.pt")
    model.to(device)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open source: {source}")

    frame_times = deque(maxlen=smoothing_window)
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=device, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        now = time.perf_counter()
        frame_times.append(now - prev_time)
        prev_time = now
        avg_frame_time = sum(frame_times) / len(frame_times)
        smoothed_fps = 1.0 / avg_frame_time

        cv2.putText(frame, f"FPS: {smoothed_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    source = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run(source=source)