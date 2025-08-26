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


def gen_cam(image, mask):
    # create a heatmap from the Grad-CAM mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = (1 - 0.5) * heatmap + 0.5 * image
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    img = io.imread("img.png")
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_input(img)

    model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
    target_layer = model.blocks[-1].norm1

    grad_cam = GradCam(model, target_layer)
    mask = grad_cam(inputs)
    result = gen_cam(img, mask)

    # save Grad-CAM result
    cv2.imwrite('result.jpg', result)

    # save original resized image
    original_image = np.uint8(255 * img)
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('original_image.jpg', original_image_bgr)

