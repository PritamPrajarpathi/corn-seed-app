from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "Random"
VALIDATION_DIR = BASE_DIR / "dataset" / "validation"
IMAGE_SIZE = (224, 224)
SEED = 42


def load_image(path: Path) -> np.ndarray:
    image = load_img(path, target_size=IMAGE_SIZE)
    array = img_to_array(image) / 255.0
    return np.array(array, dtype=np.float32)


def evaluate_autoencoder(model_path: Path):
    image_paths = []
    for path in DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            image_paths.append(path)

    if not image_paths:
        raise FileNotFoundError(f"No image files found in {DATA_DIR}")

    indices = list(range(len(image_paths)))
    random.Random(SEED).shuffle(indices)
    split_index = max(2, int(len(indices) * 0.8))
    test_paths = [image_paths[i] for i in indices[split_index:]]

    print(f"\nEvaluating autoencoder: {model_path.name}")
    X_test = np.stack([load_image(path) for path in test_paths])
    model = tf.keras.models.load_model(model_path, compile=False)
    reconstructions = model.predict(X_test, batch_size=16, verbose=0)

    mse = np.mean((X_test - reconstructions) ** 2)
    mae = np.mean(np.abs(X_test - reconstructions))
    accuracy_percent = max(0.0, 100.0 * (1.0 - mae))

    print(f"Test images: {len(test_paths)}")
    print(f"Test MSE: {mse:.6f}")
    print(f"Test MAE: {mae:.6f}")
    print(f"Mean pixel error on 0-255 scale: {mae * 255:.2f}")
    print(f"Reconstruction accuracy: {accuracy_percent:.2f}%")


def evaluate_classifier(model_path: Path):
    if not VALIDATION_DIR.exists():
        raise FileNotFoundError(f"Validation directory not found: {VALIDATION_DIR}")

    print(f"\nEvaluating classifier: {model_path.name}")
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator = datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
    )

    model = tf.keras.models.load_model(model_path, compile=False)
    predictions = model.predict(generator, verbose=0)
    y_true = generator.classes
    y_pred = np.argmax(predictions, axis=1)
    accuracy = np.mean(y_true == y_pred) * 100.0

    print(f"Validation images: {len(y_true)}")
    print(f"Validation accuracy: {accuracy:.2f}%")


def main():
    evaluate_autoencoder(BASE_DIR / "corn_model_random_autoencoder.h5")
    evaluate_classifier(BASE_DIR / "corn_model_random.h5")
    evaluate_classifier(BASE_DIR / "corn_model.h5")


if __name__ == "__main__":
    main()
