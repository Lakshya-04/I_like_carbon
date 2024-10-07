import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# Step 1: Feature Extraction Function
def extract_features(image):
    """
    Extract features from the image for anomaly detection.
    Features could include color histograms, texture properties, etc.
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate color histogram (for simplicity, we use grayscale histograms)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    
    # Normalize the histogram
    hist = hist / hist.sum()
    
    # Texture features using Gray-Level Co-occurrence Matrix (GLCM)
    glcm = greycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    
    # Combine features into a single feature vector
    features = np.hstack([hist, contrast, dissimilarity, homogeneity, energy, correlation])
    
    return features

# Step 2: Prepare Training Data
def load_images_from_folder(folder_path):
    """
    Load images from the specified folder and extract features.
    """
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(extract_features(img))
    return np.array(images)

# Load training images (good carbon brushes)
training_folder = 'path/to/good_carbon_brush_images'
training_features = load_images_from_folder(training_folder)

# Step 3: Normalize the Features
scaler = StandardScaler()
training_features_scaled = scaler.fit_transform(training_features)

# Step 4: Train Isolation Forest Model
model = IsolationForest(contamination='auto', random_state=42)
model.fit(training_features_scaled)

# Step 5: Detect Anomalies in New Carbon Brushes
def detect_anomaly(image, model, scaler):
    """
    Detect if the carbon brush is good or defective.
    """
    features = extract_features(image)
    features_scaled = scaler.transform([features])
    is_anomaly = model.predict(features_scaled)  # -1 for anomaly, 1 for normal
    return is_anomaly

# Example usage
new_image_path = 'path/to/new_carbon_brush_image.jpg'
new_image = cv2.imread(new_image_path)

if new_image is not None:
    result = detect_anomaly(new_image, model, scaler)
    if result == -1:
        print("Reject the brush")
    else:
        print("Accept the brush")
else:
    print("Error loading image!")
