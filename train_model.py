import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# --- CONFIGURATION ---
DATASET_DIR = 'dataset'  # Path to your dataset folder
IMG_SIZE = (224, 224)    # MobileNet expects 224x224
BATCH_SIZE = 32
EPOCHS = 10
CLASS_MODE = 'categorical'

def train_fish_model():
    # 1. Data Augmentation & Loading
    # MobileNetV3 expects pixel values in [-1, 1] or specific preprocessing
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
        validation_split=0.2, # Use 20% of data for validation
        horizontal_flip=True,
        rotation_range=20
    )

    train_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE,
        subset='validation'
    )

    # 2. Build the Model (Transfer Learning)
    # Load MobileNetV3 without the top classification layer (include_top=False)
    base_model = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers so we don't destroy pre-trained features
    base_model.trainable = False

    # Add custom layers for our specific fish classes
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # 3. Compile and Train
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # 4. Save the Model and Class Names
    model.save('fish_model.h5')
    
    # Save class indices (e.g., {'Sea Bass': 0, 'Trout': 1}) to use in the web app
    with open('class_names.txt', 'w') as f:
        f.write(str(train_generator.class_indices))
        
    print("Model saved as 'fish_model.h5' and classes saved to 'class_names.txt'")

if __name__ == "__main__":
    # Create dataset folder if it doesn't exist just to warn user
    if not os.path.exists(DATASET_DIR):
        print(f"Error: '{DATASET_DIR}' folder not found. Please create it and add fish images.")
    else:
        train_fish_model()
