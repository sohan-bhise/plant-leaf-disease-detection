# =============================================================
# PLANT LEAF DISEASE DETECTION — Demo Website Backend
# Flask app: serves the site, runs live ResNet50 inference,
# and exposes the RF/KNN/ResNet50 comparison data.
# =============================================================
#
# INSTALL FIRST: pip install flask torch torchvision pillow pandas

import os
import io
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
import cv2
from torchvision import transforms, models
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# ---- CONFIG — same paths as your training scripts ----
PROJECT_DIR = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject'
SEGMENTED_ROOT = os.path.join(PROJECT_DIR, 'plantvillage dataset', 'segmented')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
RESNET_WEIGHTS = os.path.join(MODEL_DIR, 'resnet50_leaf_disease.pth')

IMG_SIZE = 224
TARGET_CROPS = ('Tomato', 'Potato', 'Corn')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---- Rebuild the exact class list used during training (same sorted() order) ----
def get_target_classes(root):
    all_classes = os.listdir(root)
    return sorted([c for c in all_classes if any(crop in c for crop in TARGET_CROPS)])


CLASSES = get_target_classes(SEGMENTED_ROOT)
NUM_CLASSES = len(CLASSES)

# Human-readable split of "Crop___Disease_name" for nicer display
def prettify_label(label):
    parts = label.split('___')
    crop = parts[0].replace('_', ' ').replace('(maize)', '').strip()
    disease = parts[1].replace('_', ' ').strip() if len(parts) > 1 else ''
    return crop, disease


# ---- Rebuild model architecture (must match training script exactly) ----
def build_model(num_classes):
    model = models.resnet50(weights=None)  # no need to redownload ImageNet weights, we load our own
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model


print("Loading ResNet50 model...")
model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded on {device}. {NUM_CLASSES} classes.")

# Must match eval_transform from training exactly
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def looks_plant_like(pil_img, min_plant_fraction=0.12):
    """
    Cheap sanity gate, NOT a real out-of-distribution detector.
    Checks whether a plausible fraction of the image falls in
    green/yellow-green/brown hue ranges typical of leaves —
    catches obvious non-leaf uploads (people, objects, screenshots)
    before they get a confident false diagnosis from the classifier.
    A real leaf photo with an unusual background could still fail
    this check — treat it as a coarse filter, not a guarantee.
    """
    img = np.array(pil_img.resize((128, 128)))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Hue ~25-95 covers yellow-green through green through teal-green,
    # which covers healthy leaf, dried/yellowing leaf, and most blight
    # discoloration. Low saturation/value pixels (grays, near-black,
    # near-white) are excluded from the denominator since backgrounds
    # commonly fall there and shouldn't dilute the estimate too harshly.
    lower = np.array([15, 30, 30])
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    plant_fraction = np.sum(mask > 0) / mask.size

    return plant_fraction >= min_plant_fraction, round(plant_fraction * 100, 1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception:
        return jsonify({'error': 'Could not read image file'}), 400

    is_plant, plant_pct = looks_plant_like(img)
    if not is_plant:
        return jsonify({
            'error': f"This doesn't look like a plant leaf ({plant_pct}% plant-colored pixels detected). "
                     f"Try a clearer photo of a single leaf."
        }), 200

    tensor = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top3_probs, top3_idx = torch.topk(probs, k=3)

    results = []
    for prob, idx in zip(top3_probs.cpu().numpy(), top3_idx.cpu().numpy()):
        crop, disease = prettify_label(CLASSES[idx])
        results.append({
            'label': CLASSES[idx],
            'crop': crop,
            'disease': disease if disease.lower() != 'healthy' else 'Healthy',
            'is_healthy': disease.lower() == 'healthy',
            'confidence': round(float(prob) * 100, 2)
        })

    return jsonify({'predictions': results})


@app.route('/api/comparison')
def comparison():
    """Returns RF/KNN/ResNet50 metrics for the comparison plates section."""
    data = []

    ml_csv = os.path.join(MODEL_DIR, 'ml_results_summary.csv')
    if os.path.exists(ml_csv):
        ml_df = pd.read_csv(ml_csv)
        data.extend(ml_df.to_dict('records'))

    dl_csv = os.path.join(MODEL_DIR, 'resnet50_results.csv')
    if os.path.exists(dl_csv):
        dl_df = pd.read_csv(dl_csv)
        data.extend(dl_df.to_dict('records'))

    return jsonify({'results': data})


@app.route('/api/classes')
def classes():
    """Returns the class list grouped by crop, for the overview section."""
    grouped = {'Tomato': [], 'Potato': [], 'Corn': []}
    for cls in CLASSES:
        crop, disease = prettify_label(cls)
        for key in grouped:
            if key.lower() in crop.lower():
                grouped[key].append(disease if disease else 'Healthy')
    return jsonify(grouped)


@app.route('/modelimg/<filename>')
def model_image(filename):
    """Serves confusion matrix PNGs directly from your models/ folder."""
    return send_from_directory(MODEL_DIR, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
