import re

import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import Saliency
from captum.attr import IntegratedGradients
from captum.attr import GuidedGradCam
from captum.attr import GuidedBackprop
from captum.attr import InputXGradient
from captum.attr import NoiseTunnel
from captum.attr import DeepLift
from captum.attr import GradientShap
from captum.attr import visualization as viz

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = models.inception_v3(pretrained=True)
model.to(device)
model = model.eval()

labels_path = os.getenv("HOME") + '/.torch/models/imagenet_class_index.json'
with open(labels_path) as json_data:
    idx_to_labels = json.load(json_data)

transform = transforms.Compose([
    transforms.Resize((299, 299), antialias=True),
    transforms.ToTensor()
])

transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

img = Image.open('./data/demo_images/ILSVRC2012_val_00015410.JPEG')
img_np = np.array(img)
transformed_img = transform(img)
input = transformed_img / 127.5 - 1.0
input = input.unsqueeze(0)
input = input.to(device)
print(input.shape)

output = model(input)
output = F.softmax(output, dim=1)
prediction_score, pred_label_idx = torch.topk(output, 1)

pred_label_idx.squeeze_()
predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
# print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')
# plt.imshow(transformed_img.squeeze().cpu().detach().numpy().transpose(1 , 2, 0))
# plt.show()


default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                 [(0, '#ffffff'),
                                                  (0.25, '#000000'),
                                                  (1, '#000000')], N=256)

saliency_methods = ['Gradient', 'Smooth Grad', 'Input X Gradient', 'Guided GradCAM',
                    'Guided Backpropagation', 'Integrated Gradients', 'DeepLift', 'GradientShap']
for i in range(len(saliency_methods)):
    saliency_method = saliency_methods[i]
    if saliency_method == 'Gradient':
        saliency = Saliency(model)
        attributions = saliency.attribute(input, target=pred_label_idx)
    elif saliency_method == 'Smooth Grad':
        saliency = Saliency(model)
        noise_tunnel = NoiseTunnel(saliency)
        attributions = noise_tunnel.attribute(input, nt_type='smoothgrad', nt_samples=10, target=pred_label_idx)
    elif saliency_method == 'Input X Gradient':
        input_x_gradient = InputXGradient(model)
        attributions = input_x_gradient.attribute(input, target=pred_label_idx)
    elif saliency_method == 'Guided GradCAM':
        guided_gradcam = GuidedGradCam(model, model.Mixed_7c)
        attributions = guided_gradcam.attribute(input, target=pred_label_idx)
    elif saliency_method == 'Guided Backpropagation':
        guided_backprop = GuidedBackprop(model)
        attributions = guided_backprop.attribute(input, target=pred_label_idx)
    elif saliency_method == 'Integrated Gradients':
        integrated_gradients = IntegratedGradients(model)
        attributions = integrated_gradients.attribute(input, target=pred_label_idx)
    elif saliency_method == 'DeepLift':
        deep_lift = DeepLift(model)
        attributions = deep_lift.attribute(input, target=pred_label_idx)
    elif saliency_method == 'GradientShap':
        rand_img_dist = torch.cat([input * 0, input * 1])
        gradient_shap = GradientShap(model)
        attributions = gradient_shap.attribute(input,
                                                  n_samples=50,
                                                  stdevs=0.0001,
                                                  baselines=rand_img_dist,
                                                  target=pred_label_idx)

    _ = viz.visualize_image_attr(np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='negative')

    _[0].savefig(saliency_method + '.png')



