# gradcam-ViT

An interactive dashboard to explore a subset of the **ImageNet** dataset and analyze the predictions of a **Vision Transformer (ViT)** using **Grad-CAM**.

## Features

- **Interactive visualization of ImageNet-100 images**  
  - Images are projected into a 2D space using dimensionality reduction (**t-SNE** or **UMAP**) on the `CLS` token extracted from the model.  
  - This creates clusters that roughly correspond to the 100 different classes, allowing exploration of similarities and misclassifications.

- **Image-level explanations**  
  - Click on any image to view its classification explanation using **[Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)**, highlighting the regions that contributed most to the model's decision.


## Dashboard Demo


https://github.com/user-attachments/assets/7f67ee14-ba7a-4001-8da0-a0b956f2e09c

## Misclassification Examples

By using the dashboard, we can clearly see different types of misclassifications:

**Dataset issues** – In some cases, the dataset itself can be misleading (e.g., when an image contains multiple objects).
<p align="center">
<img width="750" alt="example_2" src="https://github.com/user-attachments/assets/31255a72-d098-488b-b6ed-6a415bb9d3d8" />
</p>

**Model mistakes** – Sometimes the model is simply wrong, for example:
<p align="center">
  <img width="280" alt="example_1" src="https://github.com/user-attachments/assets/e4bea9e8-710f-440a-a146-39b19a87f847" />
</p>

## Installation

You can run the dashboard in two ways: using Docker or manually setting up a Python environment.

### Option 1: Using Docker

Build and run the Docker container:

```bash
git clone https://github.com/fbizza/gradcam-ViT
cd gradcam-ViT
docker build -t gradcam-vit .
docker run -p 8050:8050 gradcam-vit
```

### Option 2: Manual Setup

```bash
git clone https://github.com/fbizza/gradcam-ViT
cd gradcam-ViT
python -m venv venv
source venv/bin/activate (# On Windows venv\Scripts\activate)
pip install -r requirements.txt
python main.py
```



