# =============================================================
# PLANT LEAF DISEASE DETECTION — Week 3: ResNet50 Transfer Learning
# PyTorch + GPU (RTX 4050)
# =============================================================
#
# INSTALL FIRST (if not already):
#   pip install torchvision scikit-learn pandas matplotlib seaborn tqdm pillow

import os
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix, classification_report)

# ---- CONFIG ----
SEGMENTED_ROOT = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject\plantvillage dataset\segmented'
MODEL_DIR = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject\models'
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_CROPS = ('Tomato', 'Potato', 'Corn')
IMG_SIZE = 224  # ResNet50's expected input size
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_STATE = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ---- STEP 1: Build file list ----
def get_target_classes(root):
    all_classes = os.listdir(root)
    return sorted([c for c in all_classes if any(crop in c for crop in TARGET_CROPS)])


def build_filelist(root, classes):
    filepaths = []
    labels = []
    for cls in classes:
        cls_path = os.path.join(root, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepaths.append(os.path.join(cls_path, fname))
                labels.append(cls)
    return filepaths, labels


# ---- STEP 2: Custom Dataset ----
class LeafDataset(Dataset):
    def __init__(self, filepaths, labels, class_to_idx, transform=None):
        self.filepaths = filepaths
        self.labels = [class_to_idx[l] for l in labels]
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ---- STEP 3: Transforms (augmentation on train only, NOT on val/test) ----
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ---- STEP 4: Train/Val/Test split (80/10/10, stratified) ----
def prepare_data():
    classes = get_target_classes(SEGMENTED_ROOT)
    filepaths, labels = build_filelist(SEGMENTED_ROOT, classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # First split: 80% train, 20% temp
    train_fp, temp_fp, train_lb, temp_lb = train_test_split(
        filepaths, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE
    )
    # Second split: split the 20% temp into 10% val, 10% test
    val_fp, test_fp, val_lb, test_lb = train_test_split(
        temp_fp, temp_lb, test_size=0.5, stratify=temp_lb, random_state=RANDOM_STATE
    )

    print(f"Train: {len(train_fp)} | Val: {len(val_fp)} | Test: {len(test_fp)}")

    train_ds = LeafDataset(train_fp, train_lb, class_to_idx, transform=train_transform)
    val_ds = LeafDataset(val_fp, val_lb, class_to_idx, transform=eval_transform)
    test_ds = LeafDataset(test_fp, test_lb, class_to_idx, transform=eval_transform)

    # Handle class imbalance: weighted sampler for training (oversample rare classes)
    class_counts = pd.Series(train_lb).value_counts()
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[l] for l in train_lb]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, classes, class_to_idx


# ---- STEP 5: Build ResNet50 model (transfer learning) ----
def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze early layers (keep pretrained ImageNet features), only fine-tune the later layers + new head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():  # unfreeze last residual block for fine-tuning
        param.requires_grad = True

    # Replace final classification layer for our 17 classes
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model.to(device)


# ---- STEP 6: Training loop ----
def train_model(model, train_loader, val_loader, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        start = time.time()

        # --- Train ---
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{num_epochs} [{elapsed:.1f}s] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")

    model.load_state_dict(best_model_wts)
    return model, history


# ---- STEP 7: Evaluate on test set ----
def evaluate(model, test_loader, classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    macro_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"\n{'='*60}\nRESNET50 TEST RESULTS\n{'='*60}")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('ResNet50 — Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'ResNet50_confusion_matrix.png'), dpi=150)
    plt.close()

    return {'model': 'ResNet50', 'accuracy': acc, 'macro_f1': macro_f1,
            'macro_precision': macro_precision, 'macro_recall': macro_recall}


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'resnet50_training_history.png'), dpi=150)
    plt.close()


# ---- MAIN ----
if __name__ == '__main__':
    train_loader, val_loader, test_loader, classes, class_to_idx = prepare_data()

    print(f"\nBuilding ResNet50 for {len(classes)} classes...")
    model = build_model(num_classes=len(classes))

    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    model, history = train_model(model, train_loader, val_loader, NUM_EPOCHS)

    plot_training_history(history)

    results = evaluate(model, test_loader, classes)

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'resnet50_leaf_disease.pth'))
    pd.DataFrame([results]).to_csv(os.path.join(MODEL_DIR, 'resnet50_results.csv'), index=False)

    print("\n=== DONE — model + results saved to models/ folder ===")
