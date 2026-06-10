import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(input_path):
    """Return image paths while preserving a stable order."""
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def estimate_background_lab(image_lab):
    h, w = image_lab.shape[:2]
    border = max(8, int(min(h, w) * 0.04))
    samples = np.concatenate(
        [
            image_lab[:border, :, :].reshape(-1, 3),
            image_lab[-border:, :, :].reshape(-1, 3),
            image_lab[:, :border, :].reshape(-1, 3),
            image_lab[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples, axis=0)


def build_foreground_mask(image, threshold=None):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    bg_lab = estimate_background_lab(lab)
    bg_l, _, bg_b = bg_lab

    delta_l = (lab[:, :, 0] - bg_lab[0]) * 0.35
    delta_a = lab[:, :, 1] - bg_lab[1]
    delta_b = lab[:, :, 2] - bg_lab[2]
    distance = np.sqrt(delta_l * delta_l + delta_a * delta_a + delta_b * delta_b)
    distance_u8 = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if bg_l > 95:
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        lab_b = lab[:, :, 2]

        yellow_hue = hue <= 45
        enough_color = saturation >= 25
        yellow_lab = lab_b >= max(132, bg_b + 3)
        pale_yellow_lab = (lab_b >= max(136, bg_b + 8)) & (saturation >= 12)
        mask = (
            ((yellow_hue & enough_color & yellow_lab) | pale_yellow_lab)
            & (value >= 45)
        ).astype(np.uint8) * 255
    else:
        if threshold is None:
            otsu_threshold, _ = cv2.threshold(
                distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            threshold = max(18, int(otsu_threshold * 0.75))

        _, mask = cv2.threshold(distance_u8, int(threshold), 255, cv2.THRESH_BINARY)

    kernel_3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_3, iterations=1)
    return mask


def split_touching_objects(mask, min_area, peak_ratio=0.62):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_masks = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted = contour - np.array([[[x, y]]])
        cv2.drawContours(roi_mask, [shifted], -1, 255, -1)

        distance = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        if distance.max() <= 0:
            output_masks.append((x, y, roi_mask))
            continue

        _, distance_fg = cv2.threshold(distance, peak_ratio * distance.max(), 255, 0)
        distance_fg = distance_fg.astype(np.uint8)

        erode_size = 5 if min(w, h) > 24 else 3
        erode_iterations = 2 if min(w, h) > 36 else 1
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size, erode_size)
        )
        eroded_fg = cv2.erode(roi_mask, erode_kernel, iterations=erode_iterations)

        distance_components, _ = cv2.connectedComponents(distance_fg)
        eroded_components, _ = cv2.connectedComponents(eroded_fg)
        sure_fg = eroded_fg if eroded_components > distance_components else distance_fg
        component_count, markers = cv2.connectedComponents(sure_fg)

        if component_count <= 2:
            output_masks.append((x, y, roi_mask))
            continue

        sure_bg = cv2.dilate(roi_mask, np.ones((3, 3), np.uint8), iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        watershed_input = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(watershed_input, markers)

        for label in range(2, markers.max() + 1):
            object_mask = np.zeros_like(roi_mask)
            object_mask[markers == label] = 255
            if cv2.countNonZero(object_mask) >= min_area:
                output_masks.append((x, y, object_mask))

    return output_masks


def contour_from_mask(object_mask, offset_x, offset_y):
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour + np.array([[[offset_x, offset_y]]])


def square_crop(image, bbox, padding, output_size, background_bgr):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    side = int(max(w, h) + 2 * padding)
    side = max(side, 2)

    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    right = left + side
    bottom = top + side

    image_h, image_w = image.shape[:2]
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - image_w)
    pad_bottom = max(0, bottom - image_h)

    left = max(0, left)
    top = max(0, top)
    right = min(image_w, right)
    bottom = min(image_h, bottom)

    crop = image[top:bottom, left:right]
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv2.copyMakeBorder(
            crop,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=tuple(int(v) for v in background_bgr),
        )

    interpolation = cv2.INTER_AREA if crop.shape[0] > output_size else cv2.INTER_CUBIC
    return cv2.resize(crop, (output_size, output_size), interpolation=interpolation)


def output_folder_for_image(image_path, input_root, output_root):
    image_path = Path(image_path)
    input_root = Path(input_root)
    output_root = Path(output_root)

    if input_root.is_file():
        return output_root

    relative_parent = image_path.parent.relative_to(input_root)
    return output_root / relative_parent


def crop_seeds_from_image(
    image_path,
    input_root,
    output_root,
    size,
    padding,
    min_area,
    max_area,
    threshold,
    split_touching,
    debug,
):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"SKIP: Could not read {image_path}")
        return 0

    mask = build_foreground_mask(image, threshold=threshold)
    background_bgr = np.median(image.reshape(-1, 3), axis=0)

    if split_touching:
        object_masks = split_touching_objects(mask, min_area=min_area)
        contours = [
            contour_from_mask(object_mask, offset_x, offset_y)
            for offset_x, offset_y, object_mask in object_masks
        ]
        contours = [contour for contour in contours if contour is not None]
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_dir = output_folder_for_image(image_path, input_root, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > 4.5:
            continue

        boxes.append((x, y, w, h))

    boxes.sort(key=lambda box: (box[1], box[0]))

    for index, box in enumerate(boxes, start=1):
        crop = square_crop(
            image,
            box,
            padding=padding,
            output_size=size,
            background_bgr=background_bgr,
        )
        output_path = output_dir / f"{stem}_seed_{index:03d}.jpg"
        cv2.imwrite(str(output_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    if debug:
        debug_dir = output_root / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{stem}_mask.png"), mask)
        preview = image.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / f"{stem}_boxes.jpg"), preview)

    print(f"{image_path}: extracted {len(boxes)} seeds")
    return len(boxes)


def process_dataset(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    image_paths = list_images(input_path)

    if not image_paths:
        raise SystemExit(f"No images found in {input_path}")

    total = 0
    for image_path in image_paths:
        total += crop_seeds_from_image(
            image_path=image_path,
            input_root=input_path,
            output_root=output_path,
            size=args.size,
            padding=args.padding,
            min_area=args.min_area,
            max_area=args.max_area,
            threshold=args.threshold,
            split_touching=not args.no_split_touching,
            debug=args.debug,
        )

    print(f"Done. Extracted {total} seeds into {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop individual corn seeds from large photos into square 224x224 images."
    )
    parser.add_argument(
        "-i",
        "--input",
        default="raw_photos",
        help="Input image, folder, or class-folder dataset. Default: raw_photos",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="dataset/train",
        help="Output folder for cropped seeds. Default: dataset/train",
    )
    parser.add_argument("--size", type=int, default=224, help="Final crop size in pixels.")
    parser.add_argument(
        "--padding",
        type=int,
        default=18,
        help="Padding around each detected seed before resizing.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=250,
        help="Minimum detected seed area in source-image pixels.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=None,
        help="Optional maximum detected seed area in source-image pixels.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Manual foreground threshold from 0-255. Leave unset for auto.",
    )
    parser.add_argument(
        "--no-split-touching",
        action="store_true",
        help="Disable watershed splitting for kernels that touch each other.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write detection masks and boxed previews to OUTPUT/_debug.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    process_dataset(parse_args())
