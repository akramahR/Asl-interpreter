import tensorflow as tf
from src.inference import test_model_on_video
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('../models/asl_gesture_recognition_model.keras')

# Load the label encoder classes
label_encoder = LabelEncoder()
label_classes = np.load('../models/label_classes.npy')
label_encoder.classes_ = label_classes

# Path to the test video
test_video_path = '../videos2/APPLE(2)_0.mp4'

# Test the model
predicted_gesture = test_model_on_video(model, test_video_path, label_encoder)
print(f'Predicted gesture: {predicted_gesture}')
