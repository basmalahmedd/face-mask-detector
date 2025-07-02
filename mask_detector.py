import cv2 as cv
import numpy as np
from keras.models import load_model

# Load the trained mask detection model
model_path = "mask_detector.h5"
maskNet = load_model(model_path)

# Load Haar cascade for face detection
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Preprocess full frame for prediction
def process_frame(frame):
    resized = cv.resize(frame, (224, 224))  # Resize to model's input size
    resized = resized.astype("float32") / 255.0
    resized = np.expand_dims(resized, axis=0)  # Shape: (1, 224, 224, 3)
    return resized

# Start webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run prediction on full frame
    frame_input = process_frame(frame)
    prediction = maskNet.predict(frame_input)
    prob = float(prediction[0][0])
    label = "Mask" if prob > 0.5 else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

    # Detect faces just to draw rectangle (not for prediction)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle and label over the detected face
        cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv.putText(frame, f"{label} ({prob:.2f})", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show output
    cv.imshow("Face Mask Detection", frame)

    # Exit on ESC
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
