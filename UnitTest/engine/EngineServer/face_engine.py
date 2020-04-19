import torch
from torchvision import transforms
import cv2
import numpy as np

from PIL import Image


def l2_norm(input, axis=1):
    norm = np.linalg.norm(input, ord=2, axis=axis)
    output = ((input.T) / norm).T
    return norm, output


class FaceRecognise:
    def __init__(self, pth_path="./mobilefacenet_deep/rel.pth", gpu_id=0):

        self.mean = [127.5 / 255, 127.5 / 255, 127.5 / 255]
        self.std = [79.6875 / 255, 79.6875 / 255, 79.6875 / 255]
        self.size = (96, 96)
        self.pth_path = pth_path
        self.gpu_id = gpu_id
        self.preprocess_transform = transforms.Compose([
            # transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        torch.cuda.set_device(self.gpu_id)
        self.model = torch.load(pth_path)
        self.model.eval()

    def __call__(self, img, boxes):
        imgs_inputs = self.image_transform(img, boxes)
        out = self.model(imgs_inputs)
        features = out[list(out.keys())[-1]].detach().numpy()
        return features

    def image_transform(self, img, boxes):
        im_face = img[boxes[0]:boxes[2], boxes[1]:boxes[3]]
        im_resize = cv2.resize(im_face, self.size)
        im_rgb = cv2.cvtColor(im_resize, cv2.COLOR_BGR2RGB)
        imgs_inputs = self.preprocess_transform(im_rgb)
        imgs_inputs.unsqueeze_(0)
        return imgs_inputs

    def l2_norm(self, input, axis=1):
        norm = np.linalg.norm(input, ord=2, axis=axis)
        output = ((input.T) / norm).T
        return norm, output


if __name__ == '__main__':
    img1 = cv2.imread('/home/user/DHP/test/A.jpg')
    img2 = cv2.imread('/home/user/DHP/test/B.jpg')
    img3 = cv2.imread('/home/user/DHP/test/C.jpg')
    face_reco = FaceRecognise()
    a = face_reco(img1, (0, 0, 378, 438))
    b = face_reco(img2, (0, 0, 338, 364))
    c = face_reco(img3, (0, 0, 312, 316))
    _, A = l2_norm(a)
    _, B = l2_norm(b)
    _, C = l2_norm(c)
    print('Jack  vs Jack ', np.dot(A[0, :], B[0, :]))
    print('Jack  vs jun ', np.dot(A[0, :], C[0, :]))
