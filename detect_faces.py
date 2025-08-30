import cv2
import torch
from facenet_pytorch import MTCNN

# Initialize MTCNN face detector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

# Mosaic function
def mosaic(img, scale=0.05):
    """Apply mosaic effect (pixelation) to an image region."""
    h, w = img.shape[:2]
    # shrink
    small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    # enlarge back to original
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) → RGB (MTCNN expects RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is not None:
        for (x1, y1, x2, y2) in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Extract ROI (face region)
            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size != 0:  # check if region is valid
                # Apply Mosaic filter
                face_mosaic = mosaic(face_roi, scale=0.07)

                # Replace original face with pixelated version
                frame[y1:y2, x1:x2] = face_mosaic

            # Draw bounding box (green border)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Face Mosaic with Borders", frame)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
