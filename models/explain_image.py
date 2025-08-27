import os
import timm
import torch
from skimage import io
from models.gradcam import GradCam
import numpy as np
import cv2


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

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    cam = alpha * heatmap + (1 - alpha) * image
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)



def generate_gradcam_images(input_image_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    img = io.imread(input_image_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_input(img)

    model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    target_layer = model.blocks[-1].norm1
    grad_cam = GradCam(model, target_layer)

    mask = grad_cam(inputs)
    gradcam_img = gen_cam(img, mask)


    gradcam_path = os.path.join(output_dir, 'gradcam.jpg')
    cv2.imwrite(gradcam_path, cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR))

    original_path = os.path.join(output_dir, 'original_image.jpg')
    cv2.imwrite(original_path, cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2BGR))



if __name__ == "__main__":
    generate_gradcam_images("test_img.png", "./output")
