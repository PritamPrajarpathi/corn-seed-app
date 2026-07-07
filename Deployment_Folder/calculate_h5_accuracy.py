"""
Calculate accuracy for the saved corn Keras model.

Expected test data format:
  test_folder/
    Broken/
      image1.jpg
    Damage/
      image2.jpg
    healthy/
      image3.jpg

Examples:
  python calculate_h5_accuracy.py
  python calculate_h5_accuracy.py --data raw_photos
  python calculate_h5_accuracy.py --model corn_model.h5 --classes classes.json --data raw_photos
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


DEFAULT_LABEL_ALIASES = {
    "broken": "broken",
    "break": "broken",
    "damage": "discolored",
    "damaged": "discolored",
    "discolored": "discolored",
    "silkcut": "discolored",
    "healthy": "healthy",
    "good": "healthy",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent


def load_label_map(classes_path: Path) -> dict[int, str]:
    with classes_path.open("r", encoding="utf-8") as file:
        class_indices = json.load(file)

    return {int(index): str(label) for label, index in class_indices.items()}


def normalize_label(label: str, aliases: dict[str, str]) -> str:
    cleaned = label.strip().lower().replace(" ", "_").replace("-", "_")
    return aliases.get(cleaned, cleaned)


def parse_aliases(alias_values: list[str]) -> dict[str, str]:
    aliases = dict(DEFAULT_LABEL_ALIASES)
    for value in alias_values:
        if "=" not in value:
            raise ValueError(f"Invalid alias '{value}'. Use the format folder_name=model_class.")
        source, target = value.split("=", 1)
        aliases[normalize_label(source, {})] = normalize_label(target, {})
    return aliases


def collect_images(data_dir: Path, aliases: dict[str, str]) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []

    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        label = normalize_label(class_dir.name, aliases)
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((image_path, label))

    if not samples:
        raise ValueError(f"No images were found under {data_dir}.")

    return samples


def preprocess_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = ImageOps.fit(image, image_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image_array


def predict_images(
    model,
    samples: list[tuple[Path, str]],
    label_map: dict[int, str],
    image_size: tuple[int, int],
    batch_size: int,
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        batch_images = np.stack([preprocess_image(path, image_size) for path, _ in batch])
        probabilities = model.predict(batch_images, verbose=0)
        predicted_indices = np.argmax(probabilities, axis=1)

        for (_, actual_label), predicted_index in zip(batch, predicted_indices):
            predicted_label = label_map[int(predicted_index)]
            y_true.append(actual_label)
            y_pred.append(normalize_label(predicted_label, DEFAULT_LABEL_ALIASES))

    return y_true, y_pred


def print_report(y_true: list[str], y_pred: list[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("Actual and predicted label counts do not match.")
    if not y_true:
        raise ValueError("No labels were evaluated.")

    correct = sum(actual == predicted for actual, predicted in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    labels = sorted(set(y_true) | set(y_pred))
    confusion = defaultdict(Counter)

    for actual, predicted in zip(y_true, y_pred):
        confusion[actual][predicted] += 1

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct: {correct}/{len(y_true)}")
    print()
    print("Per-class accuracy:")
    for label in labels:
        total = sum(confusion[label].values())
        class_correct = confusion[label][label]
        class_accuracy = class_correct / total * 100 if total else 0.0
        print(f"  {label}: {class_accuracy:.2f}% ({class_correct}/{total})")

    print()
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("actual\\pred".ljust(14) + "".join(label.rjust(14) for label in labels))
    for actual in labels:
        row = actual.ljust(14) + "".join(str(confusion[actual][predicted]).rjust(14) for predicted in labels)
        print(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate corn_model.h5 accuracy on a labeled image folder.")
    parser.add_argument("--model", type=Path, default=SCRIPT_DIR / "corn_model.h5", help="Path to the Keras .h5 model.")
    parser.add_argument("--classes", type=Path, default=SCRIPT_DIR / "classes.json", help="Path to classes.json.")
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "raw_photos", help="Folder with one subfolder per class.")
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="Map a folder label to a model label, for example --alias Damage=discolored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required. Install it with: pip install tensorflow-cpu") from exc

    aliases = parse_aliases(args.alias)
    label_map = load_label_map(args.classes)
    samples = collect_images(args.data, aliases)

    model = tf.keras.models.load_model(args.model)
    y_true, y_pred = predict_images(
        model=model,
        samples=samples,
        label_map=label_map,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
    )
    print_report(y_true, y_pred)


if __name__ == "__main__":
    main()
