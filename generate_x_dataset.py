import os
import shutil
import easyocr


import argparse


def process_images(input_folder: str, dataset_name: str):
    # also check for Updated folder inside input_folder
    updated_folder = os.path.join(input_folder, "Updated")

    # destination dirs
    dest_dir_images = f"data/{dataset_name}/train/images"
    dest_dir_labels = f"data/{dataset_name}/train/labels"

    os.makedirs(dest_dir_images, exist_ok=True)
    os.makedirs(dest_dir_labels, exist_ok=True)

    # setup easyocr
    print("=> Loading EasyOCR model (en)")
    reader = easyocr.Reader(["en"])
    print("=> EasyOCR model loaded successfully")

    all_files = []

    if os.path.isdir(input_folder):
        all_files += [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.endswith(".bmp")
        ]

    if os.path.isdir(updated_folder):
        all_files += [
            os.path.join(updated_folder, f)
            for f in os.listdir(updated_folder)
            if f.endswith(".bmp")
        ]
        # Also check Updated/dm
        dm_folder = os.path.join(updated_folder, "dm")
        if os.path.isdir(dm_folder):
            all_files += [
                os.path.join(dm_folder, f)
                for f in os.listdir(dm_folder)
                if f.endswith(".bmp")
            ]

    print(f"Found {len(all_files)} images to process")

    for count, img_path in enumerate(all_files, 1):
        filename = os.path.basename(img_path)
        dest_img_path = os.path.join(dest_dir_images, filename)
        label_filename = filename.replace(".bmp", ".bmp.txt")
        dest_label_path = os.path.join(dest_dir_labels, label_filename)

        # Copy image
        shutil.copy2(img_path, dest_img_path)

        # run easyocr
        results = reader.readtext(img_path)

        with open(dest_label_path, "w", encoding="utf-8") as f:
            for bbox, text, conf in results:
                if len(text.strip()) == 0:
                    continue
                # easyocr bbox format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                # we need [x_min, y_min, x_max, y_max] for dataloader?
                # Actually, wait. Let's look at `paddleocr` format: x1, y1, x2, y2 = bbox
                # And in dataset generator _det(): annotations.append(dict(points=[[x1, y1],[x2, y1],[x2, y2],[x1, y2]]))
                # So we should convert easyocr polygon to [xmin, ymin, xmax, ymax]
                xs = [point[0] for point in bbox]
                ys = [point[1] for point in bbox]
                xmin = int(min(xs))
                ymin = int(min(ys))
                xmax = int(max(xs))
                ymax = int(max(ys))

                # dataset ground-truth bounding box list
                f.write(f"{text}\t[{xmin}, {ymin}, {xmax}, {ymax}]\n")

        if count % 10 == 0:
            print(f"Processed {count}/{len(all_files)} images...")

    print("=> Finished generating dataset!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate OCR dataset pseudo-labels using EasyOCR"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing raw .bmp images",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="custom_dataset",
        help="Name of the dataset to be created in data/ folder",
    )
    args = parser.parse_args()

    process_images(args.input_folder, args.dataset_name)
