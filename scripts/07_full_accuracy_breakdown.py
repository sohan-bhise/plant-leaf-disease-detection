# =============================================================
# FULL ACCURACY BREAKDOWN — 21 accuracies per model, for the report
# 17 per-class (recall) + 3 per-crop (subset accuracy) + 1 overall
# =============================================================
#
# Recomputes exact numbers from your saved models and the SAME
# train/test splits used during training (same random_state),
# so these numbers are reproducible and match your original results.

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score

# ---- CONFIG — same paths as your other scripts ----
PROJECT_DIR = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject'
FEATURES_CSV = os.path.join(PROJECT_DIR, 'features.csv')
SEGMENTED_ROOT = os.path.join(PROJECT_DIR, 'plantvillage dataset', 'segmented')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

TARGET_CROPS = ('Tomato', 'Potato', 'Corn')
RANDOM_STATE = 42
IMG_SIZE = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_target_classes(root):
    all_classes = os.listdir(root)
    return sorted([c for c in all_classes if any(crop in c for crop in TARGET_CROPS)])


CLASSES = get_target_classes(SEGMENTED_ROOT)


def crop_of(label):
    for crop in TARGET_CROPS:
        if crop in label:
            return crop
    return 'Unknown'


def build_breakdown(y_true_labels, y_pred_labels, model_name):
    """y_true_labels / y_pred_labels are lists of the actual string class names."""
    rows = []

    # 17 per-class accuracies (recall)
    for cls in CLASSES:
        mask = [t == cls for t in y_true_labels]
        n = sum(mask)
        if n == 0:
            acc = None
        else:
            correct = sum(1 for t, p in zip(y_true_labels, y_pred_labels) if t == cls and p == cls)
            acc = correct / n
        rows.append({'model': model_name, 'level': 'class', 'name': cls,
                     'accuracy': acc, 'n_samples': n})

    # 3 per-crop accuracies (subset accuracy — full disease-level correctness
    # among that crop's own test samples)
    for crop in TARGET_CROPS:
        idx = [i for i, t in enumerate(y_true_labels) if crop_of(t) == crop]
        n = len(idx)
        if n == 0:
            acc = None
        else:
            correct = sum(1 for i in idx if y_true_labels[i] == y_pred_labels[i])
            acc = correct / n
        rows.append({'model': model_name, 'level': 'crop', 'name': crop,
                     'accuracy': acc, 'n_samples': n})

    # 1 overall accuracy
    overall = accuracy_score(y_true_labels, y_pred_labels)
    rows.append({'model': model_name, 'level': 'overall', 'name': 'ALL',
                 'accuracy': overall, 'n_samples': len(y_true_labels)})

    return rows


# ---------------------------------------------------------------
# RF and KNN — reproduce the exact same test split from features.csv
# ---------------------------------------------------------------
def evaluate_ml_models():
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=['label']).values
    y_raw = df['label'].values

    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    y = le.transform(y_raw)

    # SAME split call as 03_train_rf_knn.py — must match exactly to reproduce the same test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_test_scaled = scaler.transform(X_test)
    y_test_labels = le.inverse_transform(y_test)

    all_rows = []

    rf = joblib.load(os.path.join(MODEL_DIR, 'random_forest.joblib'))
    rf_pred = le.inverse_transform(rf.predict(X_test_scaled))
    all_rows.extend(build_breakdown(list(y_test_labels), list(rf_pred), 'RandomForest'))

    knn = joblib.load(os.path.join(MODEL_DIR, 'knn.joblib'))
    knn_pred = le.inverse_transform(knn.predict(X_test_scaled))
    all_rows.extend(build_breakdown(list(y_test_labels), list(knn_pred), 'KNN'))

    return all_rows


# ---------------------------------------------------------------
# ResNet50 — reproduce the exact same test split from file paths
# ---------------------------------------------------------------
def build_filelist(root, classes):
    filepaths, labels = [], []
    for cls in classes:
        cls_path = os.path.join(root, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(cls_path, fname))
                labels.append(cls)
    return filepaths, labels


def build_resnet_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model


def evaluate_resnet():
    filepaths, labels = build_filelist(SEGMENTED_ROOT, CLASSES)

    # SAME two-step split as 04_train_resnet50.py — reproduces the same test set
    train_fp, temp_fp, train_lb, temp_lb = train_test_split(
        filepaths, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )
    val_fp, test_fp, val_lb, test_lb = train_test_split(
        temp_fp, temp_lb, test_size=0.5, stratify=temp_lb, random_state=RANDOM_STATE
    )

    model = build_resnet_model(len(CLASSES))
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'resnet50_leaf_disease.pth'), map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preds = []
    with torch.no_grad():
        for fp in test_fp:
            img = Image.open(fp).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            output = model(tensor)
            pred_idx = output.argmax(1).item()
            preds.append(CLASSES[pred_idx])

    return build_breakdown(test_lb, preds, 'ResNet50')


if __name__ == '__main__':
    print("Evaluating RF and KNN...")
    ml_rows = evaluate_ml_models()

    print("Evaluating ResNet50 (this re-runs inference on ~2400 test images, may take a minute)...")
    resnet_rows = evaluate_resnet()

    all_rows = ml_rows + resnet_rows
    result_df = pd.DataFrame(all_rows)
    result_df['accuracy_pct'] = (result_df['accuracy'] * 100).round(2)

    out_path = os.path.join(MODEL_DIR, 'full_accuracy_breakdown.csv')
    result_df.to_csv(out_path, index=False)

    # Print a clean report-ready view
    for model_name in ['RandomForest', 'KNN', 'ResNet50']:
        print(f"\n{'='*60}\n{model_name}\n{'='*60}")
        sub = result_df[result_df['model'] == model_name]

        print("\n-- Per-class accuracy (recall) --")
        for _, row in sub[sub['level'] == 'class'].iterrows():
            print(f"  {row['name']:<55} {row['accuracy_pct']:>6.2f}%  (n={int(row['n_samples'])})")

        print("\n-- Per-crop accuracy (subset, full disease-level) --")
        for _, row in sub[sub['level'] == 'crop'].iterrows():
            print(f"  {row['name']:<55} {row['accuracy_pct']:>6.2f}%  (n={int(row['n_samples'])})")

        print("\n-- Overall accuracy --")
        overall_row = sub[sub['level'] == 'overall'].iloc[0]
        print(f"  {'All 17 classes combined':<55} {overall_row['accuracy_pct']:>6.2f}%  (n={int(overall_row['n_samples'])})")

    print(f"\n\nFull table also saved to: {out_path}")
    print("Open that CSV — it's ready to copy straight into your report table.")
