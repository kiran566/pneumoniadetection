import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(dataset_path='dataset', img_size=(224, 224), batch_size=32):
    """
    Loads train, validation and test datasets using ImageDataGenerator.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    # Training Data
    train_gen = train_datagen.flow_from_directory(
        os.path.join(dataset_path, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    # Validation Data
    val_gen = test_val_datagen.flow_from_directory(
        os.path.join(dataset_path, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary"
    )

    # Test Data
    test_gen = test_val_datagen.flow_from_directory(
        os.path.join(dataset_path, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False   # Important for evaluation metrics
    )

    return train_gen, val_gen, test_gen
