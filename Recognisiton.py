import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')  # Path to the saved model file

# Create a MediaPipe Hands instance
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Define the labels (update this to match your model's output)
labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

while cap.isOpened():
    # Read a frame from the camera
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform hand detection
    results = hands.process(image)

    # Convert the image back to BGR for display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand annotations on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the hand bounding box
            h, w, _ = image.shape
            x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
            y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
            x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
            y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)
            
            # Ensure the bounding box is square
            box_size = max(x_max - x_min, y_max - y_min)
            x_max = x_min + box_size
            y_max = y_min + box_size

            # Extract the hand image
            hand_image = image[y_min:y_max, x_min:x_max]

            # Preprocess the hand image
            hand_image = cv2.resize(hand_image, (224, 224))
            hand_image = hand_image / 255.0  # Normalize to [0, 1]
            hand_image = np.expand_dims(hand_image, axis=0)  # Add batch dimension

            # Predict the hand gesture
            prediction = model.predict(hand_image)
            predicted_label = labels[np.argmax(prediction)]

            # Display the predicted gesture
            cv2.putText(image, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Hand Detection', image)

    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
