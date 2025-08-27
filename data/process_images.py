import os
import json
import pandas as pd
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
import timm
from torchvision import transforms

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

timm_model_name = "deit_tiny_patch16_224"
timm_model = timm.create_model(timm_model_name, pretrained=True)
timm_model.eval()

timm_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

rows = []
cls_vectors = []

for img_rel_path in tqdm(image_paths, desc="Processing images"):
    img_path = os.path.join(DATA_DIR, img_rel_path)
    image = Image.open(img_path).convert("RGB")

    timm_input = timm_preprocess(image).unsqueeze(0)  # batch dimension
    with torch.no_grad():
        timm_outputs = timm_model.forward_features(timm_input)
        cls_vector = timm_outputs[:, 0, :].squeeze().tolist()  # CLS token
        cls_vectors.append(cls_vector)
        cls_vector_str = json.dumps(cls_vector)

        logits = timm_model.forward_head(timm_outputs)
        predicted_idx = logits.argmax(-1).item()
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
        "correct_classification": int(true_idx == predicted_idx),
        "cls_vector": cls_vector_str
    })

df = pd.DataFrame(rows)

tsne = TSNE(n_components=2, random_state=29)
tsne_results = tsne.fit_transform(np.array(cls_vectors))
df["tsne_1"] = tsne_results[:, 0]
df["tsne_2"] = tsne_results[:, 1]

reducer = umap.UMAP(n_components=2, random_state=29)
umap_results = reducer.fit_transform(cls_vectors)
df["umap_1"] = umap_results[:, 0]
df["umap_2"] = umap_results[:, 1]

# cls_vector in the last column
cols = [col for col in df.columns if col != "cls_vector"] + ["cls_vector"]
df = df[cols]

df.to_csv(CSV_OUTPUT_FILE, index=False)
print(f"Dataset saved to {CSV_OUTPUT_FILE}")
