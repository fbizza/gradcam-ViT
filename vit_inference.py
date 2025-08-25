import os
import json
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

DATA_DIR = "data/validation"
TARGET_IDX = 88
N_IMAGES = 30
JSON_LABELS_FILE = "data/labels.json"

with open(JSON_LABELS_FILE, "r") as f:
    idx_to_label = json.load(f)


def find_images_by_index(data_dir, target_idx, n_images):
    code = idx_to_label[str(target_idx)][0]
    class_folder = os.path.join(data_dir, code)
    if not os.path.exists(class_folder):
        return []
    images = os.listdir(class_folder)[:n_images]
    return [os.path.join(class_folder, img) for img in images]


def idx_to_label_name(idx):
    return idx_to_label.get(str(idx), ["Unknown", "Unknown"])[1]

image_paths = find_images_by_index(DATA_DIR, TARGET_IDX, N_IMAGES)
if not image_paths:
    raise ValueError(f"No img found for idx {TARGET_IDX}")


model_name = "facebook/deit-tiny-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

true_label = idx_to_label_name(TARGET_IDX)

for img_path in image_paths:
    image = Image.open(img_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_idx = outputs.logits.argmax(-1).item()
        predicted_label = idx_to_label_name(predicted_idx)

    print(f"Immagine: {img_path}")
    print(f"True index: {TARGET_IDX}")
    print(f"True label: {true_label}")
    print(f"Predicted index: {predicted_idx}")
    print(f"Predicted label: {predicted_label}")
    print("-" * 50)
