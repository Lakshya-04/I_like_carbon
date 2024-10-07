import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import joblib  # To save and load the model
import matplotlib.pyplot as plt

# Step 2: Feature Extraction Function
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

# Step 3: Load Images
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(extract_features(img))
            filenames.append(filename)
    return np.array(images), filenames

# Set folder path for training data
training_folder = '/home/malegion/scripts/good_carbon_brush_images'
training_features, _ = load_images_from_folder(training_folder)

# Step 4: Normalize the Features
scaler = StandardScaler()
training_features_scaled = scaler.fit_transform(training_features)

# Step 5: Train Isolation Forest Model
model = IsolationForest(contamination='auto', random_state=42)
model.fit(training_features_scaled)

# Save the trained model and scaler
joblib.dump(model, 'isolation_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved successfully!")

