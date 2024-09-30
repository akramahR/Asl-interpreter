import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import cv2  # For extracting FPS using OpenCV

from src.preprocessing import load_video_paths_and_labels, preprocess_video_segment

FIXED_FRAMES = 30
TARGET_SIZE = (224, 224)
TARGET_FPS = 30  # Set the target FPS to align with the model expectations

# Load the model and label encoder
model_dir = '../models'
model = load_model(os.path.join(model_dir, 'asl_gesture_recognition_model.keras'))
label_classes = np.load(os.path.join(model_dir, 'label_classes.npy'))
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# Load test videos and labels
test_videos_dir = 'tdd/fvid'
video_paths, test_labels, start_times, end_times, _ = load_video_paths_and_labels(test_videos_dir)

# Function to dynamically extract FPS from video using OpenCV
def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Extract FPS using OpenCV
    cap.release()
    return fps

# Preprocess the videos and make predictions
predictions = []
correct_count = 0  # Counter for correct predictions
total_tested = 0  # Counter for total videos tested

for i, (video_path, start_time, end_time) in enumerate(zip(video_paths, start_times, end_times)):
    # Dynamically get the FPS of the video
    video_fps = get_video_fps(video_path)

    # Preprocess each video segment, pass video_fps and target_fps to align the preprocessing
    video_segment = preprocess_video_segment(video_path, start_time, end_time, video_fps=video_fps, target_size=TARGET_SIZE, target_fps=TARGET_FPS)
    video_segment = np.expand_dims(video_segment, axis=0)  # Add batch dimension

    # Make prediction
    pred = model.predict(video_segment)
    predicted_label = np.argmax(pred, axis=-1)
    predicted_word = label_encoder.inverse_transform([predicted_label])[0]

    # Append prediction
    predictions.append(predicted_word)

    # Calculate if prediction is correct
    current_label = test_labels[i]
    if current_label == predicted_word:
        correct_count += 1

    # Increment the number of total videos tested
    total_tested += 1

    # Print video-specific results
    print(
        f"Video {i + 1}: True label = {current_label}, Predicted label = {predicted_word}, "
        f"Accuracy = {correct_count}/{total_tested} videos correct ({(correct_count / total_tested) * 100:.2f}%)"
    )

# Optional: Calculate overall accuracy
overall_accuracy = accuracy_score(test_labels, predictions)
print(f"\nOverall Test Accuracy: {overall_accuracy * 100:.2f}%")
