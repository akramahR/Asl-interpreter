import numpy as np
from src.preprocessing import preprocess_video_segment, load_data_and_labels
from src.model import create_3d_cnn_model, compile_and_train_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import os

# Load data and labels
data_dir = '../videos1'
data, labels = load_data_and_labels(data_dir)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(np.unique(labels_encoded))
labels_one_hot = to_categorical(labels_encoded, num_classes=num_classes)

# Create and train model
input_shape = data.shape[1:]
model = create_3d_cnn_model(input_shape, num_classes)
model = compile_and_train_model(model, data, labels_one_hot, epochs=20, batch_size=4)

# Create the directory if it doesn't exist
model_dir = '../models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model
model.save(os.path.join(model_dir, 'asl_gesture_recognition_model.keras'))

# Save the label encoder classes for later use in inference
np.save(os.path.join(model_dir, 'label_classes.npy'), label_encoder.classes_)

print(f"Model and label encoder classes saved in {model_dir}.")
