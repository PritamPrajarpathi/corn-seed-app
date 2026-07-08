import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import img_to_array, load_img

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "Random"
OUTPUT_DIR = BASE_DIR
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42  # for random shuffling of dataset


def collect_image_paths(data_dir: Path):
    image_paths = []

    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        image_paths.append(str(path))

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {data_dir}")

    return image_paths


def split_dataset(image_paths):
    indices = list(range(len(image_paths)))
    random.Random(SEED).shuffle(indices)

    split_index = max(2, int(len(indices) * 0.8)) #80% for training, 20% for testing
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_paths = [image_paths[i] for i in train_indices]
    test_paths = [image_paths[i] for i in test_indices]

    return train_paths, test_paths


def load_images(image_paths):
    images = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=IMAGE_SIZE)
        array = img_to_array(image) / 255.0
        images.append(array)

    return np.array(images, dtype=np.float32)


def build_autoencoder():
    inputs = layers.Input(shape=IMAGE_SIZE + (3,))

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss="mse", metrics=["mae"])
    return autoencoder


def train_model():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    image_paths = collect_image_paths(DATA_DIR)
    train_paths, test_paths = split_dataset(image_paths)

    X_train = load_images(train_paths)
    X_test = load_images(test_paths)

    print(f"Training samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")

    model = build_autoencoder()

    model.fit(
        X_train,
        X_train,
        validation_data=(X_test, X_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "corn_model_random_autoencoder.h5"
    model.save(model_path)

    with open(OUTPUT_DIR / "train_test_split.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_count": len(train_paths),
                "test_count": len(test_paths),
                "seed": SEED,
            },
            handle,
            indent=2,
        )

    print(f"Autoencoder saved to: {model_path}")


if __name__ == "__main__":
    train_model()
