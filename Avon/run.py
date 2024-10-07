import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib
import matplotlib.pyplot as plt

# Load the saved model and scaler
model = joblib.load('isolation_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature extraction function (same as before)
def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    glcm = graycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    features = np.hstack([hist, contrast, dissimilarity, homogeneity, energy, correlation])
    return features

# Step 1: Open a video stream (0 is for webcam, or use a video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract features from the frame
    features = extract_features(frame)
    
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Predict with the loaded model
    prediction = model.predict(features_scaled)
    
    # Display the result on the frame
    label = "Accepted" if prediction[0] == 1 else "Rejected"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if label == "Accepted" else (0, 0, 255), 2)
    
    # Show the frame
    cv2.imshow('Real-Time Inference', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

