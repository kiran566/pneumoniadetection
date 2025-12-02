# train.py

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import load_data
from model import build_model

dataset_path = "../dataset"
img_size = (224, 224)
batch_size = 32

train_gen, val_gen, test_gen = load_data(dataset_path, img_size, batch_size)

# Build Model 
model = build_model(input_shape=(224, 224, 3))
model.summary()  #  architecture

#  Callbacks
checkpoint = ModelCheckpoint(
    "best_model.h5", 
    monitor="val_accuracy", 
    save_best_only=True, 
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_accuracy", 
    patience=5,  # no improvement after 5 epochs
    verbose=1,
    restore_best_weights=True
)

callbacks = [checkpoint, early_stop]

#  Train Model 
epochs = 20

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks
)

# Plot Accuracy & Loss 
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
