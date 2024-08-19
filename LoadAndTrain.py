import os
import json
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Directory containing the video data and metadata
data_dir = 'videos1'

# Set a fixed number of frames
FIXED_FRAMES = 30

# Function to extract and preprocess video segment
def preprocess_video_segment(video_path, start_time, end_time, target_size=(224, 224), fps=30):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Calculate the starting and ending frames based on the provided times
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []

    # Read the frames from start_time to end_time
    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, target_size)

        # Convert the frame to an array and normalize pixel values
        frame = img_to_array(frame) / 255.0

        frames.append(frame)

    cap.release()

    # If no frames were read, initialize an empty array with the correct shape
    if len(frames) == 0:
        frames = np.zeros((FIXED_FRAMES, *target_size, 3))
    else:
        # Convert list of frames to a numpy array
        frames = np.array(frames)

        # Ensure the video segment has exactly FIXED_FRAMES frames
        if len(frames) < FIXED_FRAMES:
            # Pad with zeros if too short
            padding = np.zeros((FIXED_FRAMES - len(frames), *target_size, 3))
            frames = np.concatenate([frames, padding], axis=0)
        elif len(frames) > FIXED_FRAMES:
            # Truncate if too long
            frames = frames[:FIXED_FRAMES]

    return frames

# Function to load data and labels
def load_data_and_labels(data_dir):
    data = []
    labels = []

    for sign_dir in os.listdir(data_dir):
        sign_path = os.path.join(data_dir, sign_dir)

        for file_name in os.listdir(sign_path):
            if file_name.endswith('_metadata.json'):
                metadata_path = os.path.join(sign_path, file_name)

                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                video_path = metadata['file']
                start_time = metadata['start_time']
                end_time = metadata['end_time']
                label = metadata['clean_text']

                # Preprocess video segment
                video_segment = preprocess_video_segment(video_path, start_time, end_time)

                # Append to data and labels
                data.append(video_segment)
                labels.append(label)

    return np.array(data), np.array(labels)


# Load data and labels
data, labels = load_data_and_labels(data_dir)

print(f"Loaded {len(data)} video segments with labels.")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense

# Function to create the 3D CNN model
def create_3d_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

# Assume labels are categorical, encode them
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Encode the labels to integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Convert labels to one-hot encoding
num_classes = len(np.unique(labels_encoded))
labels_one_hot = to_categorical(labels_encoded, num_classes=num_classes)

# Create the model
input_shape = data.shape[1:]  # (num_frames, 224, 224, 3)
model = create_3d_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the entire dataset
model.fit(data, labels_one_hot, epochs=10, batch_size=4)

# Save the trained model in the .keras format
model.save('asl_gesture_recognition_model.keras')
