import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt

# Step 2: Feature Extraction Function
def extract_features(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate color histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    # Texture features using Gray-Level Co-occurrence Matrix (GLCM)
    glcm = graycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    # Combine features into a single feature vector
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
            filenames.append(filename)  # Save the filename for later display
    return np.array(images), filenames

# Set the folder path for training data (replace with local paths)
training_folder = '/home/malegion/scripts/good_carbon_brush_images'
training_features, _ = load_images_from_folder(training_folder)

# Step 4: Normalize the Features
scaler = StandardScaler()
training_features_scaled = scaler.fit_transform(training_features)

# Step 5: Train Isolation Forest Model
model = IsolationForest(contamination='auto', random_state=42)
model.fit(training_features_scaled)

# Step 6: Testing on New Dataset (replace with local paths)
testing_folder = '/home/malegion/scripts/testing_dataset'
testing_features, testing_filenames = load_images_from_folder(testing_folder)

# Normalize the testing features
testing_features_scaled = scaler.transform(testing_features)

# Predict using the model
predictions = model.predict(testing_features_scaled)

# Initialize counters
total_brushes = len(predictions)
accepted_brushes = np.sum(predictions == 1)
rejected_brushes = np.sum(predictions == -1)

print(f"Total brushes: {total_brushes}")
print(f"Accepted brushes: {accepted_brushes}")
print(f"Rejected brushes: {rejected_brushes}")

# Display rejected brushes
for i in range(len(predictions)):
    if predictions[i] == -1:  # If rejected
        print(f"Rejected brush: {testing_filenames[i]}")
        img_path = os.path.join(testing_folder, testing_filenames[i])
        img = cv2.imread(img_path)
        # Display the image using matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Rejected Brush: {testing_filenames[i]}")
        plt.show()

