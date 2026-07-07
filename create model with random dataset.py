import json
import random
from math import ceil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "Random"
OUTPUT_DIR = BASE_DIR 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

CLASS_NAMES = ["broken", "discolored", "healthy"]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def collect_image_samples(data_dir: Path):
    image_paths = []
    labels = []

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        name = path.name.lower()
        parent_name = path.parent.name.lower()

        if "broken" in name:
            label = "broken"
        elif "discolored" in name:
            label = "discolored"
        elif "healthy" in name:
            label = "healthy"
        elif parent_name in CLASS_TO_INDEX:
            label = parent_name
        else:
            continue

        image_paths.append(str(path))
        labels.append(label)

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {data_dir}")

    return image_paths, labels


def split_dataset(image_paths, labels):
    indices = list(range(len(image_paths)))
    random.Random(SEED).shuffle(indices)

    split_index = max(2, int(len(indices) * 0.8))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    return train_paths, train_labels, val_paths, val_labels


def load_dataset(image_paths, labels):
    images = []
    encoded_labels = []

    for image_path, label in zip(image_paths, labels):
        image = load_img(image_path, target_size=IMAGE_SIZE)
        array = img_to_array(image)
        array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
        images.append(array)
        encoded_labels.append(CLASS_TO_INDEX[label])

    return np.array(images, dtype=np.float32), np.array(encoded_labels, dtype=np.int32)


def train_model():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    image_paths, labels = collect_image_samples(DATA_DIR)
    train_paths, train_labels, val_paths, val_labels = split_dataset(image_paths, labels)

    X_train, y_train = load_dataset(train_paths, train_labels)
    X_val, y_val = load_dataset(val_paths, val_labels)

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Classes: {CLASS_NAMES}")

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED)

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMAGE_SIZE + (3,),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMAGE_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_generator,
        steps_per_epoch=ceil(len(X_train) / BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        verbose=1,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_generator,
        steps_per_epoch=ceil(len(X_train) / BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=2,
        verbose=1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "corn_model_random.h5"
    model.save(model_path)

    with open(OUTPUT_DIR / "classes.json", "w", encoding="utf-8") as handle:
        json.dump(CLASS_TO_INDEX, handle, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Classes saved to: {OUTPUT_DIR / 'classes.json'}")


if __name__ == "__main__":
    train_model()
