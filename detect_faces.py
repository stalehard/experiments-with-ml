import cv2
import torch
from facenet_pytorch import MTCNN

# Initialize MTCNN face detector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

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
                # Apply Gaussian Blur to the face
                blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)

                # Replace the original face with blurred version
                frame[y1:y2, x1:x2] = blurred_face

            # Draw bounding box (green border)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Face Blur with Borders", frame)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
