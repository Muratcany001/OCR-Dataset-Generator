import os
import shutil
import cv2
import easyocr
import argparse
import albumentations as A
import numpy as np


def augment_dataset(input_folder: str, dataset_name: str, num_augments: int = 5):
    dest_dir_images = f"data/{dataset_name}/train/images"
    dest_dir_labels = f"data/{dataset_name}/train/labels"

    os.makedirs(dest_dir_images, exist_ok=True)
    os.makedirs(dest_dir_labels, exist_ok=True)

    print("=> Loading EasyOCR model (en)")
    reader = easyocr.Reader(["en"])
    print("=> EasyOCR model loaded successfully")

    all_files = []
    if os.path.isdir(input_folder):
        all_files += [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.endswith((".bmp", ".jpg", ".png", ".jpeg"))
        ]

    # Include Updated if any
    updated_folder = os.path.join(input_folder, "Updated")
    if os.path.isdir(updated_folder):
        all_files += [
            os.path.join(updated_folder, f)
            for f in os.listdir(updated_folder)
            if f.endswith((".bmp", ".jpg", ".png", ".jpeg"))
        ]
        dm_folder = os.path.join(updated_folder, "dm")
        if os.path.isdir(dm_folder):
            all_files += [
                os.path.join(dm_folder, f)
                for f in os.listdir(dm_folder)
                if f.endswith((".bmp", ".jpg", ".png", ".jpeg"))
            ]

    print(f"Found {len(all_files)} images to process")

    # Define augmentation pipeline
    transform = A.Compose(
        [
            A.Affine(
                rotate=(-10, 10),
                shear={"x": (-5, 5), "y": (-5, 5)},
                scale=(0.9, 1.1),
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT,
                cval=(255, 255, 255),
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=3, p=0.2),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["texts"]),
    )

    for count, img_path in enumerate(all_files, 1):
        filename = os.path.basename(img_path)
        base_name, ext = os.path.splitext(filename)

        # 1. Process original image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # EasyOCR requires RGB or BGR, it handles BGR array when reading directly or file path.
        # Using file path for Reader is safer
        results = reader.readtext(img_path)

        bboxes = []
        texts = []

        h, w = img.shape[:2]

        for bbox, text, conf in results:
            if len(text.strip()) == 0:
                continue
            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]
            xmin = max(0, int(min(xs)))
            ymin = max(0, int(min(ys)))
            xmax = min(w, int(max(xs)))
            ymax = min(h, int(max(ys)))

            if xmin >= xmax or ymin >= ymax:
                continue

            bboxes.append([xmin, ymin, xmax, ymax])
            texts.append(text)

        # Save Original
        dest_img_path = os.path.join(dest_dir_images, f"{base_name}_orig{ext}")
        dest_label_path = os.path.join(dest_dir_labels, f"{base_name}_orig{ext}.txt")

        cv2.imwrite(dest_img_path, img)
        with open(dest_label_path, "w", encoding="utf-8") as f:
            for text, box in zip(texts, bboxes):
                f.write(
                    f"{text}\t[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]\n"
                )

        # 2. Generate Augmentations
        if len(bboxes) > 0:
            for i in range(num_augments):
                try:
                    augmented = transform(image=img, bboxes=bboxes, texts=texts)
                    aug_img = augmented["image"]
                    aug_bboxes = augmented["bboxes"]
                    aug_texts = augmented["texts"]

                    aug_img_path = os.path.join(
                        dest_dir_images, f"{base_name}_aug_{i}{ext}"
                    )
                    aug_label_path = os.path.join(
                        dest_dir_labels, f"{base_name}_aug_{i}{ext}.txt"
                    )

                    cv2.imwrite(aug_img_path, aug_img)
                    with open(aug_label_path, "w", encoding="utf-8") as f:
                        for text, box in zip(aug_texts, aug_bboxes):
                            # Ensure bounds
                            x_min = max(0, int(box[0]))
                            y_min = max(0, int(box[1]))
                            x_max = min(w, int(box[2]))
                            y_max = min(h, int(box[3]))
                            if x_min < x_max and y_min < y_max:
                                f.write(
                                    f"{text}\t[{x_min}, {y_min}, {x_max}, {y_max}]\n"
                                )
                except Exception as e:
                    print(f"Augmentation failed for {filename} augment {i}: {e}")

        if count % 5 == 0:
            print(f"Processed {count}/{len(all_files)} images...")

    print("=> Finished generating augmented dataset!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand dataset using albumentations")
    parser.add_argument(
        "--input_folder", type=str, required=True, help="Path to raw images"
    )
    parser.add_argument("--dataset_name", type=str, default="x_augmented_dataset")
    parser.add_argument(
        "--num_augments", type=int, default=5, help="Number of augmentations per image"
    )
    args = parser.parse_args()

    augment_dataset(args.input_folder, args.dataset_name, args.num_augments)
