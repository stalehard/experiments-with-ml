import cv2
import sys


# Mosaic function
def mosaic(img, scale=0.05):
    """Apply mosaic effect (pixelation) to an image region."""
    h, w = img.shape[:2]
    # shrink
    small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
    # enlarge back to original
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

while True:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    # Run a model
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_top_left = int(detections[0, 0, i, 3] * frame_width)
            y_top_left = int(detections[0, 0, i, 4] * frame_height)
            x_bottom_right  = int(detections[0, 0, i, 5] * frame_width)
            y_bottom_right  = int(detections[0, 0, i, 6] * frame_height)

            x1, y1, x2, y2 = map(int, [x_top_left, y_top_left, x_bottom_right, y_bottom_right])

            # Extract ROI (face region)
            face_roi = frame[y1:y2, x1:x2]

            if face_roi.size != 0:  # check if region is valid
                # Apply Mosaic filter
                face_mosaic = mosaic(face_roi, scale=0.07)

                # Replace original face with pixelated version
                frame[y1:y2, x1:x2] = face_mosaic

            # Draw bounding box (green border)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

source.release()
cv2.destroyWindow(win_name)
