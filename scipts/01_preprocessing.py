import cv2
import os
import numpy as np

def enhance_brightness(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y = clahe.apply(y)
    merged = cv2.merge((y, cr, cb))
    return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

def sharpen_image(frame):
    kernel = np.array([[0, -1, 0],
              [-1, 5,-1],
              [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

def denoise(frame):
    return cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

def is_blurry(frame, threshold=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


video_path = "test40.mp4"
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = max(1, int(fps / 2))

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:

        # Skip very blurry frames
        if is_blurry(frame):
            frame_count += 1
            continue

        # Apply preprocessing pipeline
        frame = enhance_brightness(frame)
        #frame = denoise(frame)
        frame = sharpen_image(frame)

        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        saved_count += 1

    frame_count += 1

print(f"Total frames saved: {saved_count}")

cap.release()
cv2.destroyAllWindows()
