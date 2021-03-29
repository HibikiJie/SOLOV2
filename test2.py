import torch
from torch import nn
from utiles.matrix_nms import matrix_nms
import torch.nn.functional as F
from models.net2 import Solo
from PIL import Image
import numpy
import cv2


class Explorer:
    def __init__(self):
        self.net = Solo()

        self.net.load_state_dict(torch.load('weights/solo2.pt', map_location='cpu'))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)
        self.net.eval()

    def __call__(self, image):
        image = self.square_image(image)
        with torch.no_grad():
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255
            (f_4_c, f_4_k), (f_8_c, f_8_k), (f_16_c, f_16_k), (f_32_c, f_32_k), (
                f_64_c, f_64_k), mask_feature = self.net(image_tensor.unsqueeze(0).to(self.device))
            maskes = []
            confidences = []
            print(f_8_c.max(), f_16_c.max(), f_32_c.max(), f_64_c.max())
            for cate, kernel in [(f_16_c, f_16_k), (f_32_c, f_32_k), (f_64_c, f_64_k)]:
                mask, confidence = self.get_mask(cate, kernel, mask_feature)

    def get_mask(self, cate, kernel, mask_feature):
        print(cate.shape, kernel.shape)
        print(cate.max())
        condition = cate[:, :, :, 0] >= 0.8
        confidence = cate[condition].reshape(-1)
        print(confidence)
        ks = kernel[condition]
        print(ks.shape)
        boxes = []
        for confi, k in zip(confidence, ks):
            print(confi, k.shape)
            mask = torch.sigmoid(F.conv2d(mask_feature, k.reshape(1, 256, 1, 1))).reshape(160, 160) * 255
            mask = mask.cpu().numpy().astype(numpy.uint8)
            # condi = mask>=128
            # mask[condi] = 255
            # mask[~condi] = 0
            cv2.imshow('a', mask)
            cv2.waitKey()
            # boxes.append(confi,)
        # exit()
        return 0, 0

    def square_image(self, image):
        background = numpy.zeros((640, 640, 3), dtype=numpy.uint8)
        # background[:, :, :] = 127
        h, w, c = image.shape
        max_len = max(w, h)
        fx = 640 / max_len
        image = cv2.resize(image, None, fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
        h, w, c = image.shape
        h_s = 320 - h // 2
        w_s = 320 - w // 2
        background[h_s:h_s + h, w_s:w_s + w] = image
        return background


if __name__ == '__main__':
    explorer = Explorer()
    image = cv2.imread('00001.jpg')
    success, maskes, socre = explorer(image)
    print(maskes.shape)
    if success:
        for mask in maskes:
            print(mask)
            mask = mask.numpy()
            # print(mask.max())
            # condition = mask > 0.38
            # mask[condition] = 1
            # mask[~condition] = 0
            mask = mask * 255
            # print(mask)
            mask = mask.astype(numpy.uint8)
            print(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow('a', mask)
            cv2.waitKey()
