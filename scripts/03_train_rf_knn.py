# =============================================================
# PLANT LEAF DISEASE DETECTION — Week 2: Train RF + KNN
# =============================================================
#
# INSTALL FIRST (if not already): pip install scikit-learn matplotlib seaborn joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)

# ---- CONFIG ----
FEATURES_CSV = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject\features.csv'
MODEL_DIR = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject\models'
import os
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_STATE = 42




def load_data():
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=['label']).values
    y_raw = df['label'].values
    return X, y_raw


def evaluate_model(name, model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"\n{'='*60}")
    print(f"{name} RESULTS")
    print('='*60)
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}   <- use this, not just accuracy")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")

    print("\nPer-class report (watch Potato_healthy specifically):")
    class_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{name} — Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, f'{name}_confusion_matrix.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")

    return {'model': name, 'accuracy': acc, 'macro_f1': macro_f1,
            'macro_precision': macro_precision, 'macro_recall': macro_recall}


def main():
    print("Loading features...")
    X, y_raw = load_data()

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Stratified split — critical given your 35x class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Scale features — matters a lot for KNN, harmless for RF
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ---- Random Forest ----
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',  # handles your imbalance directly
        random_state=RANDOM_STATE,
        n_jobs=-1  # use all CPU cores
    )
    start = time.time()
    rf.fit(X_train_scaled, y_train)
    print(f"RF train time: {time.time() - start:.1f}s")
    results.append(evaluate_model('RandomForest', rf, X_test_scaled, y_test, le))
    joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest.joblib'))

    # ---- KNN ----
    print("\nTraining KNN...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    # NOTE: sklearn's KNN has no direct class_weight param — this is a real
    # limitation worth stating honestly in your report given the imbalance.
    start = time.time()
    knn.fit(X_train_scaled, y_train)
    print(f"KNN train time: {time.time() - start:.1f}s")
    results.append(evaluate_model('KNN', knn, X_test_scaled, y_test, le))
    joblib.dump(knn, os.path.join(MODEL_DIR, 'knn.joblib'))

    # Save scaler + label encoder too — you'll need them at inference time
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.joblib'))

    # ---- Summary comparison table ----
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(MODEL_DIR, 'ml_results_summary.csv'), index=False)


if __name__ == '__main__':
    main()
