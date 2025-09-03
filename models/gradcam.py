from PIL import Image
import torch
import numpy as np

class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()
        self.feature = None
        self.gradient = None
        self.target = target  # target layer for Grad-CAM
        self._get_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = self.reshape_transform(output_grad)

        def _store_grad(grad):
            self.gradient = self.reshape_transform(grad)

        output_grad.register_hook(_store_grad)

    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    def reshape_transform(self, tensor, height=14, width=14):
        # tensor shape: (batch, tokens, channels)
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        return result

    def __call__(self, inputs):
        self.model.zero_grad()
        output = self.model(inputs)

        index = torch.argmax(output, dim=1).item()
        target = output[0, index]
        target.backward()

        gradient = self.gradient[0]  # (channels, height, width)
        weight = torch.mean(gradient, dim=(1, 2))
        feature = self.feature[0]

        cam = feature * weight[:, None, None]
        cam = torch.sum(cam, dim=0)
        cam = torch.relu(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        cam_img = (cam * 255).byte().cpu().numpy()
        cam_img = Image.fromarray(cam_img)
        cam_img = cam_img.resize((224, 224), Image.BILINEAR)
        cam_resized = torch.tensor(np.array(cam_img), dtype=torch.float32) / 255.0

        return cam_resized
