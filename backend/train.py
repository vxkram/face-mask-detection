"""Trains the face mask classifier and saves it to model/mask_detector.keras.

Dataset: download from the URL in the top-level README and extract into
backend/data/MP2_FaceMask_Dataset/ (train/ and test/ subfolders, one folder
per class: with_mask, without_mask, partial_mask).
"""
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "MP2_FaceMask_Dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    class_names = train_ds.class_names

    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

    normalize = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    base = MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    train_ds, val_ds, class_names = build_datasets()
    print("Classes:", class_names)

    model = build_model(len(class_names))
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation accuracy: {accuracy:.3f}, loss: {loss:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "mask_detector.keras"))
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    print(f"Saved model and class_names.json to {MODEL_DIR}")


if __name__ == "__main__":
    main()
