import os
import shutil
import random
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
SOURCE_ROOT = "./Rice_Leaf_Aug"   # yaha aapka main folder
FARM_1 = "data/disease/rice/farm_1"
FARM_2 = "data/disease/rice/farm_2"



IMAGES_PER_FARM = 100   # har farm me kitni images chahiye


def process_folder(src_folder, farm1_folder, farm2_folder):

    # saare images load karke shuffle kar denge
    images = [
        f for f in os.listdir(src_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) < IMAGES_PER_FARM * 2:
        print(f"[WARNING] {src_folder} me {IMAGES_PER_FARM*2} images nahi hain!")
        return

    random.shuffle(images)

    # Split unique images
    farm1_imgs = images[:IMAGES_PER_FARM]
    farm2_imgs = images[IMAGES_PER_FARM:IMAGES_PER_FARM*2]

    # copy to farm1
    for img in farm1_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(farm1_folder, img)
        )

    # copy to farm2
    for img in farm2_imgs:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(farm2_folder, img)
        )


def main():
    # ensure farm folders exist
    os.makedirs(FARM_1, exist_ok=True)
    os.makedirs(FARM_2, exist_ok=True)

    # iterate over each class folder
    for folder in os.listdir(SOURCE_ROOT):
        src_folder = os.path.join(SOURCE_ROOT, folder)
        if not os.path.isdir(src_folder):
            continue

        print(f"\nProcessing: {folder}")

        # create same folder inside farm1 and farm2
        farm1_folder = os.path.join(FARM_1, folder)
        farm2_folder = os.path.join(FARM_2, folder)

        os.makedirs(farm1_folder, exist_ok=True)
        os.makedirs(farm2_folder, exist_ok=True)

        # process unique images
        process_folder(src_folder, farm1_folder, farm2_folder)

        print(f"Completed: {folder}")


# RUN
main()

