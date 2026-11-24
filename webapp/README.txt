FIELD NOTES ON LEAF PATHOLOGY — Demo Site Setup
=================================================

1. Unzip this folder somewhere convenient, e.g.:
   C:\Users\Sohan\Desktop\Plant leaf diesese ditection\leaf_disease_webapp\

2. Install Flask and OpenCV (everything else — torch, torchvision, pillow,
   pandas — you already have installed from earlier steps):

   pip install flask opencv-python numpy

NOTE ON THE "IS THIS A LEAF" CHECK
-----------------------------------
The app now does a quick sanity check before running the classifier —
it estimates what fraction of the uploaded image is green/plant-colored,
and refuses to diagnose if that fraction is too low (catches obvious
non-leaf uploads like people, objects, screenshots — this exists because
testing showed the model will confidently misdiagnose ANY image, even
something like a photo of a car, since it has no built-in "not a leaf"
option). This is a coarse color heuristic, not real object detection —
it can still be fooled by green-dominant non-leaf images, and could
reject an unusual but genuine leaf photo (e.g. heavily backlit, or a
leaf mostly covered in white/gray fungal growth). Mention this as a
known limitation if asked in your presentation, don't present it as
a solved problem.

3. IMPORTANT — check the paths in app.py match your actual folders:

   PROJECT_DIR = r'C:\Users\Sohan\Desktop\Plant leaf diesese ditection\PlantDiseaseProject'

   This must point to the same PlantDiseaseProject folder that has your
   'models' folder (with resnet50_leaf_disease.pth, ml_results_summary.csv,
   resnet50_results.csv, and the confusion matrix PNGs) and your
   'plantvillage dataset/segmented' folder.

4. Run the app from the leaf_disease_webapp folder:

   python app.py

5. It'll print something like:
   Running on http://127.0.0.1:5000

   Open that URL in your browser (Chrome/Edge). This is what you'll
   demo live in front of the class — the browser tab, not a code editor.

TROUBLESHOOTING
---------------
- "No such file" errors on startup: double check PROJECT_DIR in app.py
  matches your actual folder name exactly, including spelling.
- Predictions look wrong/random: make sure resnet50_leaf_disease.pth
  is the actual trained weights file from your training run, not empty
  or from an interrupted run.
- Confusion matrix images not showing in Comparison Plates section:
  confirm the PNG filenames in your models/ folder are exactly
  RandomForest_confusion_matrix.png, KNN_confusion_matrix.png,
  ResNet50_confusion_matrix.png (matches what the training scripts saved).

BEFORE YOUR LIVE DEMO
----------------------
Test with 3-5 REAL photos (not from PlantVillage) of tomato/potato/corn
leaves before presenting — phone camera photos, different lighting and
backgrounds than PlantVillage's lab conditions. This checks the model
actually generalizes and lets you pick images you trust for the live
demo, rather than finding out it struggles in front of your class.
