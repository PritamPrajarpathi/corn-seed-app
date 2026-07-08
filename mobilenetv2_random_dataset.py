import json
import random
import shutil
from pathlib import Path

from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "Random"
OUTPUT_DIR = BASE_DIR
SPLIT_DIR = OUTPUT_DIR / "temp_random_split"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42


def collect_image_paths(data_dir: Path):
    image_paths = []
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        image_paths.append(path)

    if not image_paths:
        raise FileNotFoundError(f"No supported images found in {data_dir}")

    return image_paths


def infer_label(image_path: Path) -> str:
    name = image_path.name.lower()
    if "broken" in name:
        return "broken"
    if "discolored" in name:
        return "discolored"
    return "healthy"


def split_dataset(image_paths):
    grouped_paths = {}
    for image_path in image_paths:
        label = infer_label(image_path)
        grouped_paths.setdefault(label, []).append(image_path)

    train_paths = []
    test_paths = []

    for label, paths in grouped_paths.items():
        shuffled = paths[:]
        random.Random(SEED).shuffle(shuffled)
        split_index = max(1, int(len(shuffled) * 0.8))
        train_paths.extend(shuffled[:split_index])
        test_paths.extend(shuffled[split_index:])

    random.Random(SEED).shuffle(train_paths)
    random.Random(SEED + 1).shuffle(test_paths)

    return train_paths, test_paths


def prepare_split_folders(train_paths, test_paths):
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    train_dir = SPLIT_DIR / "train"
    test_dir = SPLIT_DIR / "test"

    for label in ["broken", "discolored", "healthy"]:
        (train_dir / label).mkdir(parents=True, exist_ok=True)
        (test_dir / label).mkdir(parents=True, exist_ok=True)

    for image_path in train_paths:
        label = infer_label(image_path)
        destination = train_dir / label / image_path.name
        shutil.copy2(image_path, destination)

    for image_path in test_paths:
        label = infer_label(image_path)
        destination = test_dir / label / image_path.name
        shutil.copy2(image_path, destination)

    return train_dir, test_dir


def train_model():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    image_paths = collect_image_paths(DATA_DIR)
    train_paths, test_paths = split_dataset(image_paths)
    train_dir, test_dir = prepare_split_folders(train_paths, test_paths)

    print(f"Training samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )

    val_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
    )

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
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(train_generator.num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        verbose=1,
    )

    final_val_accuracy = history.history["val_accuracy"][-1]
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "corn_model_mobilenetv2_random.h5"
    model.save(model_path)

    with open(OUTPUT_DIR / "classes.json", "w", encoding="utf-8") as handle:
        json.dump(train_generator.class_indices, handle, indent=2)

    print("Model saved to:", model_path)

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)


if __name__ == "__main__":
    train_model()