import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib
import os
import matplotlib.pyplot as plt

# Load the saved model and scaler
model = joblib.load('isolation_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature extraction function
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

# Step 1: Load images from a folder (use your folder path)
def load_images_as_video(folder_path):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder_path)):  # Sort to simulate a video sequence
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append((img, filename))
    return images

# Set folder path for testing images (use your folder path)
testing_folder = '/home/malegion/scripts/testing_dataset'
image_sequence = load_images_as_video(testing_folder)

# Simulate video stream using the image sequence
for img, filename in image_sequence:
    # Extract features from the image
    features = extract_features(img)

    # Scale the features
    features_scaled = scaler.transform([features])

    # Predict with the loaded model
    prediction = model.predict(features_scaled)

    # Display the result on the image
    label = "Accepted" if prediction[0] == 1 else "Rejected"
    color = (0, 255, 0) if label == "Accepted" else (0, 0, 255)
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the image (mimicking a video stream)
    cv2.imshow('Image Stream', img)
    
    # Wait for a short time to simulate frame rate (e.g., 100ms between frames)
    if cv2.waitKey(100) & 0xFF == ord('q'):  # Break on 'q' key press
        break

# Release resources
cv2.destroyAllWindows()

