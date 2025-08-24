import torch
from facenet_pytorch import MTCNN
import cv2
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)

# Load image with OpenCV
img = cv2.imread("files/crowd.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR -> RGB for PyTorch

# Detect faces
boxes, probs = mtcnn.detect(rgb_img)

# Draw boxes
if boxes is not None:
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(rgb_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Show image with matplotlib
plt.imshow(rgb_img)
plt.axis("off")
plt.show()
