# gradcam-ViT

An interactive dashboard to explore a subset of the **ImageNet** dataset and analyze the predictions of a **Vision Transformer (ViT)** using **Grad-CAM**.

## Features

- **Interactive visualization of ImageNet-100 images**  
  - Images are projected into a 2D space using dimensionality reduction (**t-SNE** or **UMAP**) on the `CLS` token extracted from the model.  
  - This creates clusters that roughly correspond to the 100 different classes, allowing exploration of similarities and misclassifications.

- **Image-level explanations**  
  - Click on any image to view its classification explanation using **[Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)**, highlighting the regions that contributed most to the model's decision.


## Dashboard Demo

![Dashboard Demo](path/to/screenshot.png)

## Installation

You can run the dashboard in two ways: using **Docker** or manually setting up a Python environment.

### Option 1: Using Docker

Build and run the Docker container:

```bash
git clone https://github.com/fbizza/gradcam-ViT.git
cd gradcam-ViT
docker build -t gradcam-vit .
docker run -p 8050:8050 gradcam-vit

