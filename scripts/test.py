import os
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import load_data_and_labels, load_video_paths_and_labels, preprocess_video_segment

FIXED_FRAMES = 30
TARGET_SIZE = (224, 224)
FPS = 30


# Load the model and label encoder
model_dir = '../models'
model = load_model(os.path.join(model_dir, 'asl_gesture_recognition_model.keras'))
label_classes = np.load(os.path.join(model_dir, 'label_classes.npy'))
label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

# Load test videos and labels
test_videos_dir = 'test_videos'
video_paths, test_labels, start_times, end_times = load_video_paths_and_labels(test_videos_dir)

# Preprocess the videos and make predictions
predictions = []
correct_count = 0  # Counter for correct predictions
total_tested = 0  # Counter for total videos tested

for i, (video_path, start_time, end_time) in enumerate(zip(video_paths, start_times, end_times)):
    # Preprocess each video segment
    video_segment = preprocess_video_segment(video_path, start_time, end_time)
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
        f"Accuracy = {correct_count}/{total_tested} videos correct ({(correct_count / total_tested) * 100:.2f}%)")

# Optional: Calculate overall accuracy
overall_accuracy = accuracy_score(test_labels, predictions)
print(f"\nOverall Test Accuracy: {overall_accuracy * 100:.2f}%")
