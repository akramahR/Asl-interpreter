import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, TimeDistributed, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionV3

def create_3d_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Define the I3D architecture

def I3D(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Stem Block
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu')(inputs)  # No TimeDistributed here
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same')(x)

    # Intermediate Blocks
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same')(x)

    # Further layers
    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='same')(x)

    # Classifier Block
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Final Dense Layer
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x)
    return model


def create_pretrained_3d_cnn_model(input_shape, num_classes):
    model = Sequential()

    # Layer 1
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Layer 2
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Layer 3
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))  # Add dropout to reduce overfitting
    model.add(Dense(num_classes, activation='softmax'))

    return model

def create_3d_cnn_model_weight(input_shape, num_classes, use_pretrained_weights=False):
    model = I3D(input_shape, num_classes)

    # Load weights if specified
    if use_pretrained_weights:
        model.load_weights('path/to/your/flow_charades_weights.h5', by_name=True)  # Load weights by name

        # Freeze the lower layers
        for layer in model.layers[:-2]:  # Adjust the index based on how many layers you want to freeze
            layer.trainable = False

    return model


def compile_and_train_model(model, data, labels, epochs=20, batch_size=4, validation_size=0.2):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Split data into training and validation sets using stratified sampling
    data_train, data_val, labels_train, labels_val = train_test_split(
        data, labels, test_size=validation_size, stratify=labels, random_state=42
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the training data and validate on the validation set
    model.fit(data_train, labels_train, epochs=epochs, batch_size=batch_size, validation_data=(data_val, labels_val))


def compile_and_train_model_generator(model, train_generator, epochs=20, validation_generator=None):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create an EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=5,          # Number of epochs with no improvement to wait before stopping
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    # Fit the model using the data generator
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping]  # Include the early stopping callback
    )
    return model
