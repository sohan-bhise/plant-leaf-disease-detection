# =============================================================
# REAL-WORLD GENERALIZATION TEST — compares BOTH ResNet50 versions
# (segmented-trained vs color-trained) on the same real photos,
# side by side, so you can actually tell which one generalizes better.
# =============================================================
#
# SETUP:
# 1. Create this folder if it doesn't exist:
#    C:\Users\Sohan\Desktop\Plant leaf diesese ditection\real_world_test\
# 2. Put 8-10 real leaf photos in it (from Google Images / reliable
#    ag extension sites), with the TRUE crop as the filename prefix, e.g.:
#    potato_early_blight_01.jpg
#    tomato_septoria_01.jpg
#    corn_common_rust_01.jpg

import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

PROJECT_DIR = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject'
SEGMENTED_ROOT = os.path.join(PROJECT_DIR, 'plantvillage dataset', 'segmented')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

SEGMENTED_WEIGHTS = os.path.join(MODEL_DIR, 'resnet50_leaf_disease.pth')
COLOR_WEIGHTS = os.path.join(MODEL_DIR, 'resnet50_leaf_disease_color.pth')

TEST_FOLDER = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\real_world_test'

IMG_SIZE = 224
TARGET_CROPS = ('Tomato', 'Potato', 'Corn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_target_classes(root):
    all_classes = os.listdir(root)
    return sorted([c for c in all_classes if any(crop in c for crop in TARGET_CROPS)])


CLASSES = get_target_classes(SEGMENTED_ROOT)  # same 17 classes, same sorted order, either model


def build_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model


def load_model(weights_path):
    model = build_model(len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


print("Loading both models...")
model_segmented = load_model(SEGMENTED_WEIGHTS)
model_color = load_model(COLOR_WEIGHTS)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_prob, top_idx = torch.max(probs, dim=0)
    return CLASSES[top_idx], float(top_prob) * 100


if __name__ == '__main__':
    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)
        print(f"Created folder: {TEST_FOLDER}")
        print("Put your real test photos in there (see naming convention in the")
        print("comment at the top of this script), then run this again.")
        exit()

    files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print(f"No images found in {TEST_FOLDER}. Add some real leaf photos first.")
        exit()

    seg_correct, color_correct = 0, 0

    print(f"\n{'FILE':<32}{'TRUE CROP':<11}{'SEGMENTED-TRAINED':<42}{'COLOR-TRAINED':<42}")
    print("-" * 127)

    for fname in sorted(files):
        stated_crop = fname.split('_')[0].capitalize()
        img = Image.open(os.path.join(TEST_FOLDER, fname)).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        seg_label, seg_conf = predict(model_segmented, tensor)
        color_label, color_conf = predict(model_color, tensor)

        seg_crop = seg_label.split('___')[0]
        color_crop = color_label.split('___')[0]

        seg_ok = stated_crop.lower() in seg_crop.lower()
        color_ok = stated_crop.lower() in color_crop.lower()
        seg_correct += seg_ok
        color_correct += color_ok

        seg_str = f"{seg_label} ({seg_conf:.1f}%) {'OK' if seg_ok else 'X'}"
        color_str = f"{color_label} ({color_conf:.1f}%) {'OK' if color_ok else 'X'}"

        print(f"{fname:<32}{stated_crop:<11}{seg_str:<42}{color_str:<42}")

    print("-" * 127)
    n = len(files)
    print(f"\nSegmented-trained crop accuracy: {seg_correct}/{n} ({100*seg_correct/n:.1f}%)")
    print(f"Color-trained crop accuracy:     {color_correct}/{n} ({100*color_correct/n:.1f}%)")
    print("\n(This only checks CROP match from filename. Check disease-level")
    print("correctness yourself by reading the predicted labels above.)")
