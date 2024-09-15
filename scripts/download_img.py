import requests
from PIL import Image
from io import BytesIO
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser

# File paths for datasets
SAMPLED_TRAIN_PATH = os.path.join("ml24", "dataset", "sampled_train.csv")
TRAIN_PATH = os.path.join("ml24", "dataset", "train_cleaned.csv")
TEST_PATH = os.path.join("ml24", "dataset", "test.csv")
BASE_IMG_URL = "https://m.media-amazon.com/images/I/"
TRAIN_SAVEPATH = "train_images/raw"
TEST_SAVEPATH = "test_images/raw"


def download_image(img_url, save_dirpath, img_no, tot_imgs):
    """Downloads and saves an image from the given URL."""
    img_name = img_url.split("/")[-1]
    try:
        res = requests.get(img_url)
        res.raise_for_status()
        img = Image.open(BytesIO(res.content))
        img.save(os.path.join(save_dirpath, img_name))
        print(
            f"[{img_no + 1}/{tot_imgs}] {img_name} downloaded :: {len(res.content)/1e6:.2f} MB"
        )
    except Exception as e:
        print(f"Error downloading from {img_url}: {e}")


def get_images_to_download(df, save_dirpath):
    """Identifies which images need to be downloaded based on the dataset."""
    img_download = set(df.image_name.unique())
    img_present = set(os.listdir(save_dirpath))
    img_to_download = img_download - img_present
    print(
        f"Images present: {len(img_present)}; Left to download: {len(img_to_download)}"
    )
    return [BASE_IMG_URL + img for img in img_to_download]


def sample_images(df, sample_size):
    """Samples a specified number of images from the dataset and saves the sampled data."""
    if sample_size > len(df):
        raise ValueError(
            f"Sample size ({sample_size}) is greater than dataset size ({len(df)})."
        )
    sampled_df = df.sample(sample_size)
    sampled_df.to_csv(SAMPLED_TRAIN_PATH, index=False)
    return sampled_df


def concurrent_download(imgs_download, save_dirpath, workers):
    """Downloads images concurrently using ThreadPoolExecutor."""
    print(f"Saving images to: {save_dirpath}")
    tot_imgs = len(imgs_download)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for img_no, url in enumerate(imgs_download):
            executor.submit(download_image, url, save_dirpath, img_no, tot_imgs)


def ensure_save_directory_exists(save_dirpath):
    """Ensures the save directory exists, creates it if necessary."""
    if not os.path.exists(save_dirpath):
        os.makedirs(save_dirpath)


def parse_arguments():
    """Parses command-line arguments."""
    parser = ArgumentParser(description="Image downloader script")
    parser.add_argument(
        "--workers",
        "-w",
        default=8,
        type=int,
        help="Number of workers for downloading images",
    )
    parser.add_argument(
        "--dataset_type",
        "-d",
        default="train",
        choices=["train", "test"],
        help="Dataset to use (train/test)",
    )
    parser.add_argument(
        "--savepath", "-sp", default="", help="Directory to save images"
    )
    parser.add_argument(
        "--sample",
        "-s",
        default=-1,
        type=int,
        help="Number of images to sample. Default (-1) is all.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Select the dataset path based on argument
    dataset_path = TRAIN_PATH if args.dataset_type == "train" else TEST_PATH
    savepath = args.savepath
    if savepath == "":
        savepath = TRAIN_SAVEPATH if args.dataset_type == "train" else TEST_SAVEPATH
    df = pd.read_csv(dataset_path)

    # Ensure save directory exists
    ensure_save_directory_exists(savepath)

    # Sample the dataset if specified
    if args.sample > 0 and args.sample < len(df):
        if not os.path.exists(SAMPLED_TRAIN_PATH):
            df = sample_images(df, args.sample)
        else:
            df = pd.read_csv(SAMPLED_TRAIN_PATH)

    # Get images to download and start downloading concurrently
    imgs_to_download = get_images_to_download(df, savepath)
    concurrent_download(imgs_to_download, args.savepath, args.workers)


if __name__ == "__main__":
    main()
