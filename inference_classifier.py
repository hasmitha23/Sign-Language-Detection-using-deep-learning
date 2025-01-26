import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the camera (change the index if needed)
cap = cv2.VideoCapture(0)  # Change to 0 or another index if necessary

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Mapping from label index to sign language letters
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Failed to grab frame")
        break  # Exit the loop if the frame is not captured

    # Get the frame's dimensions (height, width)
    H, W, _ = frame.shape

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    # If hands are detected, draw the landmarks and connections
    if results.multi_hand_landmarks:
        # Only process the first hand if multiple hands are detected
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame,  # Image to draw
            hand_landmarks,  # Hand landmarks
            mp_hands.HAND_CONNECTIONS,  # Hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Initialize lists for storing the normalized hand landmark coordinates
        data_aux = []
        x_ = []
        y_ = []

        # Extract hand landmarks and normalize the coordinates
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y

            x_.append(x)
            y_.append(y)

        # Normalize the landmarks' coordinates by subtracting the minimum value
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))

        # Calculate the bounding box coordinates for displaying the predicted sign
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict the sign language letter using the trained model
        prediction = model.predict([np.asarray(data_aux)])

        # Map the prediction index to a sign language letter
        predicted_character = labels_dict[int(prediction[0])]

        # Draw a bounding box around the hand and display the predicted letter
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame with the predicted character
    cv2.imshow('frame', frame)

    # Wait for a key press (1 ms) to continue to the next frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
