import os
import json
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification

DATA_DIR = "images"
JSON_LABELS_FILE = "labels_dictionary.json"
CSV_OUTPUT_FILE = "dataset.csv"

with open(JSON_LABELS_FILE, "r") as f:
    idx_to_label = json.load(f)

code_to_idx = {val[0]: int(idx) for idx, val in idx_to_label.items()}

image_paths = []
for code_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, code_folder)
    if os.path.isdir(folder_path):
        for img_file in os.listdir(folder_path):
            image_paths.append(os.path.join(code_folder, img_file))

model_name = "facebook/deit-tiny-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

rows = []

for img_rel_path in tqdm(image_paths, desc="Processing images"):
    img_path = os.path.join(DATA_DIR, img_rel_path)
    image = Image.open(img_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_idx = outputs.logits.argmax(-1).item()
        predicted_label = idx_to_label[str(predicted_idx)][1]

    code = img_rel_path.split(os.sep)[0]
    true_idx = code_to_idx.get(code, -1)
    true_label = idx_to_label[str(true_idx)][1] if true_idx != -1 else "Unknown"

    rows.append({
        "img_name": img_rel_path,
        "true_idx": true_idx,
        "true_label": true_label,
        "predicted_idx": predicted_idx,
        "predicted_label": predicted_label,
        "correct_classification": int(true_idx == predicted_idx)
    })

pd.DataFrame(rows).to_csv(CSV_OUTPUT_FILE, index=False)
print(f"Dataset saved to {CSV_OUTPUT_FILE}")
