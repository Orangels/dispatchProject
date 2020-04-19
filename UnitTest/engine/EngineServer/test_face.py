# coding:utf-8
# from mobilefacenet_deep.mobilefacenet import MobileNetV1
#
# face_model = MobileNetV1(num_classes=128)
import torch
from torchvision import transforms
import cv2
import numpy as np

from PIL import Image


def l2_norm(input, axis=1):
    norm = np.linalg.norm(input, ord=2, axis=axis)
    output = ((input.T) / norm).T
    return norm, output


mean = [127.5 / 255, 127.5 / 255, 127.5 / 255]
std = [79.6875 / 255, 79.6875 / 255, 79.6875 / 255]
preprocess_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

img_A = Image.open('test/A.jpg')
img_A_tensor = preprocess_transform(img_A)
img_A_tensor.unsqueeze_(0)
pretrained_dict = torch.load('mobilefacenet_deep/rel.pth')
pretrained_dict.eval()

result = pretrained_dict(img_A_tensor)
print('FA', result[list(result.keys())[-1]].detach().numpy())
_, A = l2_norm(result[list(result.keys())[-1]].detach().numpy())

img_A = Image.open('test/B.jpg')
img_A_tensor = preprocess_transform(img_A)
img_A_tensor.unsqueeze_(0)
result = pretrained_dict(img_A_tensor)
print('FB', result[list(result.keys())[-1]].detach().numpy())
_, B = l2_norm(result[list(result.keys())[-1]].detach().numpy())

img_A = Image.open('test/C.jpg')
img_A_tensor = preprocess_transform(img_A)
img_A_tensor.unsqueeze_(0)
result = pretrained_dict(img_A_tensor)
print('FC', result[list(result.keys())[-1]].detach().numpy())
_, C = l2_norm(result[list(result.keys())[-1]].detach().numpy())
print('Jack  vs Jack ', np.dot(A[0, :], B[0, :]))
print('Jack  vs Jun ', np.dot(A[0, :], C[0, :]))
