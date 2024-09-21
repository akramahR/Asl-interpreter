from videoGenerator import *
import os

# Define constants
FIXED_FRAMES = 30

# Define constants
BATCH_SIZE = 4
TARGET_SIZE = (224, 224)
FPS = 30

# Load video paths, labels, start_times, and end_times
data_dir = '../scripts/videos'
video_paths, labels, start_times, end_times = load_video_paths_and_labels(data_dir)

######################################################
# Filter out videos with only 1 occurrence of a label
from collections import Counter
label_counts = Counter(labels)
rare_labels = [label for label, count in label_counts.items() if count < 2]
filtered_indices = [i for i, label in enumerate(labels) if label not in rare_labels]

video_paths = [video_paths[i] for i in filtered_indices]
labels = [labels[i] for i in filtered_indices]
start_times = [start_times[i] for i in filtered_indices]
end_times = [end_times[i] for i in filtered_indices]

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))

# Print out the classes the label encoder has been trained on
print("Label encoder classes:", label_encoder.classes_)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_paths, val_paths, train_labels, val_labels, train_start_times, val_start_times, train_end_times, val_end_times = train_test_split(
    video_paths, labels_encoded, start_times, end_times, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Print out training labels for debugging
print("Training labels:", train_labels)
print("Validation labels:", val_labels)

# Create data generators
train_generator = VideoDataGenerator(train_paths, train_labels, train_start_times, train_end_times, label_encoder, BATCH_SIZE, target_size=TARGET_SIZE, fps=FPS)
val_generator = VideoDataGenerator(val_paths, val_labels, val_start_times, val_end_times, label_encoder, BATCH_SIZE, target_size=TARGET_SIZE, fps=FPS)

# Create and train model
input_shape = (FIXED_FRAMES, *TARGET_SIZE, 3)
model = create_3d_cnn_model(input_shape, num_classes)
model = compile_and_train_model_generator(model, train_generator, epochs=20, validation_generator=val_generator)

# Create the directory if it doesn't exist
model_dir = '../models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model
model.save(os.path.join(model_dir, 'asl_gesture_recognition_model.keras'))

# Save the label encoder classes for later use in inference
np.save(os.path.join(model_dir, 'label_classes.npy'), label_encoder.classes_)

print(f"Model and label encoder classes saved in {model_dir}.")
