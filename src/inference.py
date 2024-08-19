import numpy as np
from src.preprocessing import preprocess_video_segment
import cv2

def preprocess_new_video(video_path, start_time=0, end_time=None, target_size=(224, 224), fps=30):
    cap = cv2.VideoCapture(video_path)
    if end_time is None:
        end_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    video_segment = preprocess_video_segment(video_path, start_time, end_time, target_size, fps)
    return np.expand_dims(video_segment, axis=0)

def test_model_on_video(model, video_path, label_encoder):
    preprocessed_video = preprocess_new_video(video_path)

    predictions = model.predict(preprocessed_video)
    predicted_label = np.argmax(predictions, axis=-1)

    # Inverse transform the predicted label to get the original class name
    return label_encoder.inverse_transform([predicted_label])[0]
