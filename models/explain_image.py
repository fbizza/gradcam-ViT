import os
import timm
import torch
import numpy as np
from PIL import Image
from matplotlib import cm
from models.gradcam import GradCam


def prepare_input(image):
    image = image.copy()
    means = np.array([0.5, 0.5, 0.5])
    stds = np.array([0.5, 0.5, 0.5])
    image -= means
    image /= stds
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    image = image[np.newaxis, ...]  # add batch dimension
    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask, alpha=0.5):
    colormap = cm.get_cmap('jet')
    heatmap = colormap(mask)[:, :, :3]  # ignore alpha channel

    cam = alpha * heatmap + (1 - alpha) * image
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def generate_gradcam_images(input_image_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_image_path).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.float32(np.array(img)) / 255
    inputs = prepare_input(img_np)

    model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    target_layer = model.blocks[-1].norm1
    grad_cam = GradCam(model, target_layer)

    mask = grad_cam(inputs)
    gradcam_img = gen_cam(img_np, mask)

    gradcam_path = os.path.join(output_dir, 'gradcam.jpg')
    Image.fromarray(gradcam_img).save(gradcam_path)

    original_path = os.path.join(output_dir, 'original_image.jpg')
    Image.fromarray(np.uint8(img_np * 255)).save(original_path)


if __name__ == "__main__":
    generate_gradcam_images("test_img.png", "./output")
