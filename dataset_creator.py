import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_directory_structure(base_path, labels):
    """
    Create directory structure for storing images.
    """
    for label in labels:
        os.makedirs(os.path.join(base_path, label), exist_ok=True)

def capture_images_for_label(base_path, label, num_images_per_label=100):
    """
    Capture images for a single label from the webcam and save them in the appropriate directory.
    """
    cap = cv2.VideoCapture(0)  # Initialize webcam
    print(f'Capturing images for label: {label}')
    for img_num in range(num_images_per_label):
        ret, frame = cap.read()
        if not ret:
            print('Failed to capture image')
            break

        cv2.imshow('Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit capturing early
            break

        img_path = os.path.join(base_path, label, f'{label}_{img_num}.jpg')
        cv2.imwrite(img_path, frame)

    print(f'Captured {num_images_per_label} images for label: {label}')
    
    cap.release()
    cv2.destroyAllWindows()

def augment_data(base_path, labels):
    """
    Augment the captured images to increase dataset size and variability.
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for label in labels:
        image_dir = os.path.join(base_path, label)
        save_to_dir = os.path.join(base_path, label)
        for img_file in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path)
            img = img.reshape((1,) + img.shape)  # Reshape for the ImageDataGenerator

            i = 0
            for batch in datagen.flow(img, batch_size=1, save_to_dir=save_to_dir, save_prefix=label, save_format='jpg'):
                i += 1
                if i > 5:
                    break

# Define base path
base_path = 'dataset'

# Interactively capture images for each label
labels = []

while True:
    label = input("Enter the label you are showing or '1' to quit: ").strip()
    if label.lower() == '1':
        break
    if label:
        labels.append(label)
        os.makedirs(os.path.join(base_path, label), exist_ok=True)
        num_images_per_label = int(input(f"Enter the number of images to capture for {label}: "))
        capture_images_for_label(base_path, label, num_images_per_label)
    else:
        print("Invalid label. Please enter a valid label.")

# Augment the captured data (optional)
augment_data(base_path, labels)

'''This file create datasets by the name 'DATASET' that will contain the images that will be used to train the model'''