import numpy as np
from src.preprocessing import preprocess_video_segment, load_video_paths_and_labels, processFramesToTensor
from src.model import create_3d_cnn_model, compile_and_train_model_generator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch
# Define constants
FIXED_FRAMES = 32

class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, start_times, end_times, fps_list, label_encoder, batch_size,
                 target_size=(224, 224), target_fps=32, augment_frame_datagen=None, apply_temporal_augmentation=False):
        self.video_paths = video_paths
        self.labels = labels
        self.start_times = start_times
        self.end_times = end_times
        self.fps_list = fps_list  # Store the list of FPS values
        self.label_encoder = label_encoder
        self.batch_size = batch_size
        self.target_size = target_size
        self.target_fps = target_fps
        self.indexes = np.arange(len(self.video_paths))
        self.augment_frame_datagen = augment_frame_datagen
        self.apply_temporal_augmentation = apply_temporal_augmentation

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_video_paths = [self.video_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        batch_start_times = [self.start_times[i] for i in batch_indexes]
        batch_end_times = [self.end_times[i] for i in batch_indexes]
        batch_fps = [self.fps_list[i] for i in batch_indexes]

        #X = np.zeros((len(batch_video_paths), FIXED_FRAMES, *self.target_size, 3))
        #y = np.zeros((len(batch_labels), len(self.label_encoder.classes_)))  # Use label_encoder classes length

        X = []
        y = []

        for i, video_path in enumerate(batch_video_paths):
            # print(f"Processing video: {video_path}, start: {batch_start_times[i]}, end: {batch_end_times[i]}, fps:{batch_fps}")
            # Preprocess video segment
            video_segment = preprocess_video_segment(video_path, batch_start_times[i], batch_end_times[i],
                                                     batch_fps[i], target_size=self.target_size, target_fps=self.target_fps)

            tensor = processFramesToTensor(video_segment)

            X.append(tensor)
            # Use the already encoded labels directly
            y.append(to_categorical(batch_labels[i], num_classes=len(self.label_encoder.classes_)))

        return X, y



    def on_epoch_end(self):
        np.random.shuffle(self.indexes)


