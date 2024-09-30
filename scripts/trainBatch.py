from src.model import I3D
from videoGenerator import *
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
FIXED_FRAMES = 32
BATCH_SIZE = 4
TARGET_SIZE = (224, 224)
FPS = 32

# Load video paths, labels, start_times, and end_times
data_dir = '../scripts/tdd/vid'
video_paths, labels, start_times, end_times, fps_list = load_video_paths_and_labels(data_dir)

######################################################
# Filter out videos with only 1 occurrence of a label
from collections import Counter
label_counts = Counter(labels)
rare_labels = [label for label, count in label_counts.items() if count < 10]
filtered_indices = [i for i, label in enumerate(labels) if label not in rare_labels]

video_paths = [video_paths[i] for i in filtered_indices]
labels = [labels[i] for i in filtered_indices]
start_times = [start_times[i] for i in filtered_indices]
end_times = [end_times[i] for i in filtered_indices]
fps_list = [fps_list[i] for i in filtered_indices]

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))


# Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_paths, val_paths, train_labels, val_labels, train_start_times, val_start_times, train_end_times, val_end_times, train_fps_list, val_fps_list = train_test_split(
    video_paths, labels_encoded, start_times, end_times, fps_list, test_size=0.2, random_state=42, stratify=labels_encoded
)


# Define spatial augmentation for frames using ImageDataGenerator
frame_datagen = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   brightness_range=[0.8, 1.2])

# Create data generators with augmentation
train_generator = VideoDataGenerator(
    video_paths=train_paths,
    labels=train_labels,
    start_times=train_start_times,
    end_times=train_end_times,
    fps_list=train_fps_list,
    label_encoder=label_encoder,
    batch_size=BATCH_SIZE,
    target_size=TARGET_SIZE,
    target_fps=FPS,
    augment_frame_datagen=frame_datagen,  # Enable spatial augmentation
    apply_temporal_augmentation=True      # Enable temporal augmentation
)

val_generator = VideoDataGenerator(
    video_paths=val_paths,
    labels=val_labels,
    start_times=val_start_times,
    end_times=val_end_times,
    fps_list=val_fps_list,
    label_encoder=label_encoder,
    batch_size=BATCH_SIZE,
    target_size=TARGET_SIZE,
    target_fps=FPS
)

for video in train_generator:
    print("asd")