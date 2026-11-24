# =============================================================
# PLANT LEAF DISEASE DETECTION — Week 1/2: Feature Extraction
# Extracts color, texture, and shape features from segmented
# leaf images for RF/KNN training. Saves to a CSV so you never
# have to re-run this expensive step more than once.
# =============================================================
#
# INSTALL FIRST (PyCharm terminal, venv active):
#   pip install opencv-python scikit-image scikit-learn pandas numpy tqdm

import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# ---- CONFIG ----
SEGMENTED_ROOT = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject\plantvillage dataset\segmented'
OUTPUT_CSV = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject\features.csv'
IMG_SIZE = 128  # resize all images to this size before extracting features

TARGET_CROPS = ('Tomato', 'Potato', 'Corn')


def get_target_classes(root):
    all_classes = os.listdir(root)
    return sorted([c for c in all_classes if any(crop in c for crop in TARGET_CROPS)])


def extract_color_features(img_hsv):
    """32-bin histogram per HSV channel, normalized. 96 features total."""
    features = []
    for i in range(3):  # H, S, V channels
        hist = cv2.calcHist([img_hsv], [i], None, [32], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)
    return features


def extract_texture_features(img_gray):
    """GLCM texture features: contrast, homogeneity, energy, correlation.
    Computed at 4 angles and averaged for rotation-robustness."""
    glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    features = []
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        vals = graycoprops(glcm, prop)
        features.append(vals.mean())
    return features


def extract_shape_features(img_gray):
    """Hu moments from the largest contour (the leaf/disease shape).
    7 features, log-scaled since Hu moments span huge ranges."""
    _, thresh = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [0.0] * 7  # blank/failed segmentation fallback

    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    hu = cv2.HuMoments(moments).flatten()

    # log-scale to compress range, handle negative/zero values safely
    hu_log = [-np.sign(h) * np.log10(abs(h) + 1e-10) for h in hu]
    return hu_log


def extract_all_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color_feats = extract_color_features(img_hsv)
    texture_feats = extract_texture_features(img_gray)
    shape_feats = extract_shape_features(img_gray)

    return color_feats + texture_feats + shape_feats


def main():
    classes = get_target_classes(SEGMENTED_ROOT)
    print(f"Extracting features from {len(classes)} classes...")

    rows = []
    labels = []

    for cls in classes:
        cls_path = os.path.join(SEGMENTED_ROOT, cls)
        img_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for fname in tqdm(img_files, desc=cls):
            img_path = os.path.join(cls_path, fname)
            feats = extract_all_features(img_path)
            if feats is not None:
                rows.append(feats)
                labels.append(cls)

    # Column names: color(96) + texture(4) + shape(7) = 107 features
    color_cols = [f'color_{i}' for i in range(96)]
    texture_cols = ['contrast', 'homogeneity', 'energy', 'correlation']
    shape_cols = [f'hu_{i}' for i in range(7)]
    all_cols = color_cols + texture_cols + shape_cols

    df = pd.DataFrame(rows, columns=all_cols)
    df['label'] = labels

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows x {len(all_cols)} features to {OUTPUT_CSV}")
    print(df['label'].value_counts())


if __name__ == '__main__':
    main()
