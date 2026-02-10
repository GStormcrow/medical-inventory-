import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import pickle

# Directory structure:
# training_data/
#   ├── aspirin/
#   │   ├── img1.jpg
#   │   ├── img2.jpg
#   ├── ibuprofen/
#   │   ├── img1.jpg
#   │   ├── img2.jpg
#   └── ...

def create_pill_recognition_model(num_classes):
    """Create CNN model for pill classification"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(224, 224, 3)),
        
        # Convolutional blocks
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model():
    """Train the pill recognition model"""
    # Load and preprocess data
    data_dir = 'training_data'
    
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create and train model
    num_classes = len(train_generator.class_indices)
    model = create_pill_recognition_model(num_classes)
    
    # Train
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5)
        ]
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/pill_recognition_model.h5')
    
    # Save class labels
    with open('models/pill_labels.pkl', 'wb') as f:
        pickle.dump(train_generator.class_indices, f)
    
    print(f"Model trained! Final accuracy: {history.history['accuracy'][-1]:.2f}")
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.2f}")


if __name__ == '__main__':
    train_model()