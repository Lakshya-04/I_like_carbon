import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib

# Load the saved model and scaler
model = joblib.load('isolation_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Optimized feature extraction function
def extract_features(image):
    hist = cv2.calcHist([image], [0], None, [64], [0, 256]).flatten()  # Fewer bins
    hist = hist / hist.sum()  # Normalize histogram
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)  # Reduce distances
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    features = np.hstack([hist, contrast, dissimilarity, homogeneity, energy, correlation])
    return features

# Open a video stream (0 is for webcam)
cap = cv2.VideoCapture(0)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (320, 240))  # Resize frame to smaller resolution
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Skip frames to process every 3rd frame (optional)
    frame_count += 1
    if frame_count % 3 != 0:
        continue

    # Extract features
    features = extract_features(gray_frame)

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict with the loaded model
    prediction = model.predict(features_scaled)

    # Display the result on the frame
    label = "Accepted" if prediction[0] == 1 else "Rejected"
    color = (0, 255, 0) if label == "Accepted" else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Show the frame
    cv2.imshow('Real-Time Inference', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

