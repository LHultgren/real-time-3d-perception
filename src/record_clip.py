import cv2

def record_clip(source=1, output="data/test_clip.mp4", seconds=10, fps=30):
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ))

    for _ in range(seconds * fps):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved to {output}")

if __name__ == "__main__":
    record_clip()