from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser

# Directory paths
PROCESSED_DIR = os.path.join("train_images", "processed")
RAW_DIR = os.path.join("train_images", "raw")


def resize_image(img_name, height, width, img_no, total_imgs, processed_dir, raw_dir):
    """Resizes and saves the image if it hasn't already been processed."""
    processed_path = os.path.join(processed_dir, img_name)
    raw_path = os.path.join(raw_dir, img_name)

    if os.path.exists(processed_path):
        print(f"{processed_path} already exists!")
        return

    try:
        img = Image.open(raw_path).resize((width, height))
        img.save(processed_path)
        print(f"[{img_no + 1}/{total_imgs}] {img_name} processed.")
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        print(f"removing {raw_path}")
        os.remove(raw_path)


def concurrent_processing(images, height, width, workers, processed_dir, raw_dir):
    """Processes images concurrently using a ThreadPoolExecutor."""
    total_imgs = len(images)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for img_no, img_name in enumerate(images):
            executor.submit(
                resize_image,
                img_name,
                height,
                width,
                img_no,
                total_imgs,
                processed_dir,
                raw_dir,
            )


def get_images_to_process(processed_dir, raw_dir):
    """Determines which images need to be processed."""
    raw_images = set(os.listdir(raw_dir))
    processed_images = set(os.listdir(processed_dir))
    images_to_process = raw_images - processed_images

    print(
        f"Already processed: {len(processed_images)}; To be processed: {len(images_to_process)}"
    )
    return images_to_process


def ensure_directory_exists(directory):
    """Ensures the directory exists, creates it if necessary."""
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory)


def parse_arguments():
    """Parses command-line arguments."""
    parser = ArgumentParser(description="Image processing script for resizing images.")
    parser.add_argument(
        "--dim",
        "-dim",
        type=int,
        default=256,
        help="Dimension (height and width) for resizing images.",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="Number of workers for concurrent processing.",
    )
    parser.add_argument(
        "--dataset_type",
        "-d",
        choices=["train", "test"],
        default="train",
        help="Dataset type (train/test).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Adjust paths based on dataset type
    processed_dir = (
        PROCESSED_DIR if args.dataset_type == "train" else "test_images/processed"
    )
    raw_dir = RAW_DIR if args.dataset_type == "train" else "test_images/raw"

    # Ensure necessary directories exist
    ensure_directory_exists(processed_dir)

    # Get list of images to process
    images_to_process = get_images_to_process(processed_dir, raw_dir)

    # Start concurrent image processing
    concurrent_processing(
        images_to_process, args.dim, args.dim, args.workers, processed_dir, raw_dir
    )


if __name__ == "__main__":
    main()
