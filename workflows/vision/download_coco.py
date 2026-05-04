"""Download COCO val2017 (images + annotations) into `data/coco/`.

Run once before the first vision experiment:
    python -m workflows.vision.download_coco
"""

import os
import subprocess
import sys

COCO_DIR = "data/coco"
IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_coco(coco_dir: str = COCO_DIR) -> None:
    os.makedirs(coco_dir, exist_ok=True)

    images_dir = os.path.join(coco_dir, "val2017")
    if not os.path.isdir(images_dir):
        print("Downloading COCO val2017 images (~800 MB)...")
        subprocess.run(["wget", "-c", IMAGES_URL, "-P", coco_dir], check=True)
        subprocess.run(
            ["unzip", "-q", os.path.join(coco_dir, "val2017.zip"), "-d", coco_dir],
            check=True,
        )
        os.remove(os.path.join(coco_dir, "val2017.zip"))

    annotations_dir = os.path.join(coco_dir, "annotations")
    if not os.path.isdir(annotations_dir):
        print("Downloading COCO annotations (~250 MB)...")
        subprocess.run(["wget", "-c", ANNOTATIONS_URL, "-P", coco_dir], check=True)
        subprocess.run(
            [
                "unzip",
                "-q",
                os.path.join(coco_dir, "annotations_trainval2017.zip"),
                "-d",
                coco_dir,
            ],
            check=True,
        )
        os.remove(os.path.join(coco_dir, "annotations_trainval2017.zip"))

    print(f"COCO val2017 ready in {coco_dir}")


if __name__ == "__main__":
    download_coco()
