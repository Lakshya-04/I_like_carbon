import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

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

# Initialize Isolation Forest model and scaler (assuming model is trained beforehand)
scaler = StandardScaler()
model = IsolationForest(contamination='auto', random_state=42)

# Step 6: Run detection on camera feed
def process_camera_feed(model, scaler, training_folder, testing_folder):
    # Load and train on the initial dataset
    training_features, _ = load_images_from_folder(training_folder)
    training_features_scaled = scaler.fit_transform(training_features)
    model.fit(training_features_scaled)

    # Open camera feed
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Here, you should apply brush detection logic
        # For simplicity, let's assume we are capturing every frame
        # In real scenarios, use object detection or template matching to detect brush in frame
        cv2.imshow('Camera Feed', frame)

        # Capture image when brush is detected (or every frame for now)
        features = extract_features(frame)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)

        if prediction == -1:
            print("Rejected brush detected")
            # Display the rejected brush
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title("Rejected Brush in Frame")
            plt.show()

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Set paths for training and testing data (replace with actual paths)
training_folder = '/home/malegion/scripts/good_carbon_brush_images'
testing_folder = '/home/malegion/scripts/testing_dataset'

# Run the camera feed processing function
process_camera_feed(model, scaler, training_folder, testing_folder)

