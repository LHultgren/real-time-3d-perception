import cv2
import time

def run(source=0):
    cap = cv2.VideoCapture(source)  # 0 = webcam,
    # run("path/till/video.mp4") for video
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open the source: {source}")

    # Track the starting high-precision timestamp to calculate FPS
    prev_time = time.perf_counter()

    while True:
        # Read the next frame from the video stream
        # ret: boolean indicating success, frame: the image array
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if video ends or webcam fails

        # Calculate FPS based on the time elapsed since the previous frame
        now = time.perf_counter()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # Overlay the FPS text on the image frame (position, font, scale, color in BGR, thickness)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam/video file hardware resources
    cap.release()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
    