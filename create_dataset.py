import os
import pickle
import mediapipe as mp
import cv2

# Initialize Mediapipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path to the dataset directory
DATA_DIR = './data'

# Initialize lists to hold data and labels
data = []
labels = []

# Process each subdirectory (class) in the data folder
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Skip files that are not directories
    if not os.path.isdir(dir_path):
        continue

    print(f"Processing class '{dir_}'...")

    # Process each image in the class directory
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)

        # Read and preprocess the image
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Unable to read image {img_full_path}. Skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Collect normalized x, y coordinates
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                # Normalize coordinates to top-left corner
                x_min, y_min = min(x_), min(y_)
                data_aux = [(x - x_min, y - y_min) for x, y in zip(x_, y_)]

                # Flatten the coordinates and add to dataset
                data.append([coord for pair in data_aux for coord in pair])
                labels.append(dir_)

# Save the data and labels to a pickle file
output_file = 'data.pickle'
with open(output_file, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created successfully and saved to '{output_file}'!")
