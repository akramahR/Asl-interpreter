import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import os
import json

FIXED_FRAMES = 30

def preprocess_video_segment(video_path, start_time, end_time, target_size=(224, 224), fps=30):
    cap = cv2.VideoCapture(video_path)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []

    while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, target_size)
        frame = img_to_array(frame) / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        frames = np.zeros((FIXED_FRAMES, *target_size, 3))
    else:
        frames = np.array(frames)
        if len(frames) < FIXED_FRAMES:
            padding = np.zeros((FIXED_FRAMES - len(frames), *target_size, 3))
            frames = np.concatenate([frames, padding], axis=0)
        elif len(frames) > FIXED_FRAMES:
            frames = frames[:FIXED_FRAMES]

    return frames


def load_data_and_labels(data_dir):
    data = []
    labels = []

    for sign_dir in os.listdir(data_dir):
        sign_path = os.path.join(data_dir, sign_dir)

        for file_name in os.listdir(sign_path):
            if file_name.endswith('_metadata.json'):
                metadata_path = os.path.join(sign_path, file_name)

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                video_path = metadata['file']
                start_time = metadata['start_time']
                end_time = metadata['end_time']
                label = metadata['clean_text']

                video_segment = preprocess_video_segment(video_path, start_time, end_time)

                data.append(video_segment)
                labels.append(label)

    return np.array(data), np.array(labels)