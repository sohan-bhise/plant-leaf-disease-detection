# =============================================================
# PLANT LEAF DISEASE DETECTION — Week 1: Dataset Setup + EDA
# LOCAL (PyCharm) VERSION — no Colab, no Drive mount
# =============================================================
#
# SETUP FIRST (do this in terminal, not in this script):
#   python -m venv venv
#   venv\Scripts\activate          (Windows)
#   source venv/bin/activate       (Mac/Linux)
#   pip install kaggle pandas matplotlib pillow numpy
#
# Then get kaggle.json (kaggle.com -> Settings -> API -> Create New Token)
# and place it at:
#   Windows: C:\Users\<you>\.kaggle\kaggle.json
#   Mac/Linux: ~/.kaggle/kaggle.json
#
# GPU CHECK — run this snippet first to know if MobileNet training
# will be usable locally later:
#   import torch; print(torch.cuda.is_available())
#   (or, if using TensorFlow: import tensorflow as tf; print(tf.config.list_physical_devices('GPU')))
# If it prints False / empty list, plan to do MobileNet training on Colab instead.

import os
import random
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---- CONFIG ----
PROJECT_DIR = os.path.join(os.getcwd(), 'PlantDiseaseProject')  # created in your current folder
os.makedirs(PROJECT_DIR, exist_ok=True)


# ---- STEP 1: Download dataset via Kaggle CLI ----
def download_dataset():
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', 'abdallahalidev/plantvillage-dataset',
        '-p', PROJECT_DIR,
        '--unzip'
    ], check=True)
    # NOTE: if this dataset slug 404s, search "plantvillage" on Kaggle's site,
    # grab the correct slug (URL after kaggle.com/datasets/), and swap it in.
    # Make sure it's the COLOR/RGB version, not segmented-only or grayscale.


# ---- STEP 2: Find the actual image folder (structure varies by mirror) ----
def locate_data_root():
    for root, dirs, filenames in os.walk(PROJECT_DIR):
        if dirs and any('___' in d or 'healthy' in d.lower() for d in dirs):
            print("Likely data root:", root)
            print("Sample classes:", dirs[:5])
            return root
    print("Couldn't auto-detect. Manually browse the extracted folder and set DATA_ROOT.")
    return None


# ---- STEP 3: Filter to target crops ----
def get_target_classes(data_root, target_crops=('Tomato', 'Potato', 'Corn')):
    all_classes = os.listdir(data_root)
    selected = [c for c in all_classes if any(crop in c for crop in target_crops)]
    print(f"Found {len(selected)} classes across target crops:")
    for c in sorted(selected):
        print(" -", c)
    return selected


# ---- STEP 4: Class distribution EDA ----
def class_distribution(data_root, classes):
    counts = {}
    for cls in classes:
        cls_path = os.path.join(data_root, cls)
        n = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        counts[cls] = n

    df = pd.DataFrame(list(counts.items()), columns=['class', 'count']).sort_values('count', ascending=False)
    print(df.to_string(index=False))
    print(f"\nTotal images: {df['count'].sum()}")
    print(f"Imbalance ratio (max/min): {df['count'].max() / df['count'].min():.2f}x")

    plt.figure(figsize=(12, 6))
    plt.barh(df['class'], df['count'])
    plt.xlabel('Image count')
    plt.title('Class distribution — Tomato / Potato / Corn (PlantVillage)')
    plt.tight_layout()
    out_path = os.path.join(PROJECT_DIR, 'class_distribution.png')
    plt.savefig(out_path)
    plt.show()
    print(f"Saved chart to {out_path}")
    return df


# ---- STEP 5: Image size check ----
def check_image_sizes(data_root, classes):
    sample_sizes = []
    for cls in random.sample(classes, min(5, len(classes))):
        cls_path = os.path.join(data_root, cls)
        files_in_cls = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in random.sample(files_in_cls, min(10, len(files_in_cls))):
            with Image.open(os.path.join(cls_path, f)) as img:
                sample_sizes.append(img.size)

    sizes_df = pd.DataFrame(sample_sizes, columns=['width', 'height'])
    print(sizes_df.describe())
    print("\nUnique sizes found:", sizes_df.drop_duplicates().shape[0])


# ---- STEP 6: Visual sanity check ----
def show_sample_images(data_root, classes):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    sample_classes = random.sample(classes, min(10, len(classes)))

    for ax, cls in zip(axes.flatten(), sample_classes):
        cls_path = os.path.join(data_root, cls)
        files_in_cls = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        sample_file = random.choice(files_in_cls)
        img = Image.open(os.path.join(cls_path, sample_file))
        ax.imshow(img)
        ax.set_title(cls, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(PROJECT_DIR, 'sample_images.png')
    plt.savefig(out_path)
    plt.show()
    print(f"Saved sample grid to {out_path}")


# ---- RUN ----
if __name__ == '__main__':
    # Comment this out after first successful run — no need to re-download every time
    download_dataset()

    DATA_ROOT = locate_data_root()
    if DATA_ROOT is None:
        # If auto-detect fails, hardcode it here after checking the folder manually, e.g.:
        # DATA_ROOT = os.path.join(PROJECT_DIR, 'PlantVillage', 'color')
        raise SystemExit("Set DATA_ROOT manually and re-run.")

    classes = get_target_classes(DATA_ROOT)
    class_distribution(DATA_ROOT, classes)
    check_image_sizes(DATA_ROOT, classes)
    show_sample_images(DATA_ROOT, classes)

    print("\n=== EDA COMPLETE ===")
