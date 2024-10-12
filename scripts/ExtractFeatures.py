from src.preprocessing import load_video_paths_and_labels
from videoGenerator import *
import torch
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.parser import load_config, parse_args
import slowfast.utils.checkpoint as cu
args = parse_args()
cfg = load_config(args)
cfg = assert_and_infer_cfg(cfg)
from slowfast.models import build_model
import os
import numpy as np
from UTILS import Identity
import pickle
from collections import Counter

# Define constants
FIXED_FRAMES = 32
BATCH_SIZE = 4
TARGET_SIZE = (224, 224)
FPS = 32

# Load video paths, labels, start_times, and end_times for training data
data_dir = 'videos'
video_paths, labels, start_times, end_times, fps_list = load_video_paths_and_labels(data_dir)

# Filter out videos with less than 10 occurrences of a label
label_counts = Counter(labels)
common_labels = [label for label, count in label_counts.items() if count > 16]
# Print the common labels
print("Common labels:", common_labels)

filtered_indices = [i for i, label in enumerate(labels) if label in common_labels]

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

'''
# Load test videos for validation
test_data_dir = 'test_videos'
test_video_paths, test_labels, test_start_times, test_end_times, test_fps_list = load_video_paths_and_labels(test_data_dir)

# Filter test videos based on the filtered rare labels
test_filtered_indices = [i for i, label in enumerate(test_labels) if label in common_labels]

test_video_paths = [test_video_paths[i] for i in test_filtered_indices]
test_labels = [test_labels[i] for i in test_filtered_indices]
test_start_times = [test_start_times[i] for i in test_filtered_indices]
test_end_times = [test_end_times[i] for i in test_filtered_indices]
test_fps_list = [test_fps_list[i] for i in test_filtered_indices]

# Encode test labels
test_labels_encoded = label_encoder.transform(test_labels)
'''

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


# Create data generators with augmentation
train_generator = VideoDataGenerator(
    video_paths=video_paths,
    labels=labels_encoded,
    start_times=start_times,
    end_times=end_times,
    fps_list=fps_list,
    label_encoder=label_encoder,
    target_size=TARGET_SIZE,
    target_fps=FPS
)

'''
val_generator = VideoDataGenerator(
    video_paths=test_video_paths,
    labels=test_labels_encoded,
    start_times=test_start_times,
    end_times=test_end_times,
    fps_list=test_fps_list,
    label_encoder=label_encoder,
    target_size=TARGET_SIZE,
    target_fps=FPS
)
'''
val_generator = VideoDataGenerator(
    video_paths=val_paths,
    labels=val_labels,
    start_times=val_start_times,
    end_times=val_end_times,
    fps_list=val_fps_list,
    label_encoder=label_encoder,
    target_size=TARGET_SIZE,
    target_fps=FPS
)

# Load the model
model = build_model(cfg)
cu.load_test_checkpoint(cfg, model)
model.head = Identity()
model.eval()
model.to(device="cuda:0")

# Initialize lists to store features and labels for training and validation sets
train_features = []
train_labels = []

# Loop over the training generator
for i, video in enumerate(train_generator):
    print('i:' + str(i))
    videoTensor = video[0].to(dtype=torch.float32)
    label = video[1]
    labelsNum = np.array(label)  # Convert labels to NumPy array

    # Get features from the model without updating the gradients
    with torch.no_grad():
        features = model(videoTensor.to(device="cuda:0"))
        featuresNum = features.cpu().numpy()  # Move features to CPU and convert to NumPy

    train_features.append(featuresNum)
    train_labels.append(label)

train_features = np.vstack(train_features)  # Stack features vertically
train_labels = np.vstack(train_labels)

trainData = np.hstack((train_features, train_labels))
np.save("trainData.npy", trainData)

val_features = []
val_labels = []

# Loop over the validation generator
for i, video in enumerate(val_generator):
    videoTensor = video[0].to(dtype=torch.float32)
    label = video[1]
    labelsNum = np.array(label)  # Convert labels to NumPy array

    # Get features from the model without updating the gradients
    with torch.no_grad():
        features = model(videoTensor.to(device="cuda:0"))
        featuresNum = features.cpu().numpy()  # Move features to CPU and convert to NumPy

    val_features.append(featuresNum)
    val_labels.append(label)

val_features = np.vstack(val_features)  # Stack features vertically
val_labels = np.vstack(val_labels)

valData = np.hstack((val_features, val_labels))
np.save("valData.npy", valData)
