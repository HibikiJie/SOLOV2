from torch.utils.data import Dataset
import os
from PIL import Image
import numpy
import torch
import cv2
from random import randint


class MyDataset(Dataset):

    def __init__(self, mode='train'):
        super(MyDataset, self).__init__()
        self.dataset = []
        mask_root = '/media/cq/data/data/tianci/PreRoundData/Annotations'
        image_root = '/media/cq/data/data/tianci/PreRoundData/JPEGImages'
        with open(f'/media/cq/data/data/tianci/PreRoundData/ImageSets/{mode}.txt') as f:
            for line in f:
                for file_name in os.listdir(f'{mask_root}/{line.strip()}'):
                    file_name = file_name[:-4]
                    mask_path = f'{mask_root}/{line.strip()}/{file_name}.png'
                    image_path = f'{image_root}/{line.strip()}/{file_name}.jpg'
                    self.dataset.append((image_path, mask_path))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        '''读取图片'''
        image_path, mask_path = self.dataset[item]
        # print(image_path)
        image = cv2.imread(image_path)
        mask = Image.open(mask_path)
        mask = numpy.array(mask)

        '''resize 图片至固定尺寸'''
        image = self.image_resize_640(image)
        mask = self.mask_resize_640(mask)

        '''制作mask和box标签'''
        # cv2.imshow('image', image)
        # cv2.imshow('labels',cv2.imread(mask_path))
        # cv2.waitKey()
        boxes = []
        maskes = []
        for i in range(1, mask.max() + 1):
            mask_a = numpy.zeros((640, 640), numpy.uint8)
            condition = mask == i
            # print(i)
            mask_a[condition] = 255
            success, box = self.out_box(mask_a)
            # print(box)
            if success:
                boxes.append(box)
                mask_a = cv2.resize(mask_a, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
                # cv2.imshow('mask', mask_a)
                #
                # cv2.waitKey()
                maskes.append(torch.from_numpy(mask_a).float() / 255)
                # cv2.imshow('mask', mask_a)
                # cv2.waitKey()

        target_40, target_32, target_24, target_16, target_12 = self.make_labels(boxes)

        '''格式转换为张量'''
        image_tensor = torch.from_numpy(image).float() / 255
        target_40 = torch.from_numpy(target_40)
        target_32 = torch.from_numpy(target_32)
        target_24 = torch.from_numpy(target_24)
        target_16 = torch.from_numpy(target_16)
        target_12 = torch.from_numpy(target_12)
        if len(maskes) == 0:
            maskes = torch.zeros(1, 160, 160).float()
        else:
            maskes = torch.stack(maskes)
        return image_tensor.permute(2, 0, 1), maskes, target_40, target_32, target_24, target_16, target_12

    def image_resize_640(self, image):
        background = numpy.zeros((640, 640, 3), dtype=numpy.uint8)
        # background[:, :, :] = randint(0, 255)
        background[:, :, :] = 127
        h, w, c = image.shape
        max_len = max(w, h)
        fx = 640 / max_len
        image = cv2.resize(image, None, fx=fx, fy=fx, interpolation=cv2.INTER_AREA)
        h, w, c = image.shape
        h_s = 320 - h // 2
        w_s = 320 - w // 2
        background[h_s:h_s + h, w_s:w_s + w] = image
        return background

    def mask_resize_640(self, image):
        background = numpy.zeros((640, 640), dtype=numpy.uint8)
        h, w = image.shape
        max_len = max(w, h)
        fx = 640 / max_len
        image = cv2.resize(image, None, fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
        h, w = image.shape
        h_s = 320 - h // 2
        w_s = 320 - w // 2
        background[h_s:h_s + h, w_s:w_s + w] = image
        return background

    def out_box(self, image):
        mask = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        ret, thresh = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY_INV)
        m = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, m, iterations=11)
        thresh = cv2.Canny(thresh, 0, 255)
        # cv2.imshow('canny', thresh)
        # cv2.waitKey()
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # print(x, y, w, h)
            if w * h >= 630 * 630 or w * h < 50:
                continue
            c_x = x + w / 2
            c_y = y + h / 2
            x1, y1, x2, y2 = c_x - w / 2, c_y - h / 2, c_x + w / 2, c_y + h / 2
            # mask = cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
            # cv2.imshow('a', mask)
            # cv2.waitKey(1)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            return True, (x1, y1, x2, y2)
        return False, (0, 0, 0, 0)
        # cv2.imshow('a', mask)
        # cv2.waitKey()
        # raise ValueError('图片没有找到目标')
        # return 1,1,1,1

    def make_labels(self, boxes):
        target_40 = numpy.zeros((40, 40, 2))
        target_32 = numpy.zeros((32, 32, 2))
        target_24 = numpy.zeros((24, 24, 2))
        target_16 = numpy.zeros((16, 16, 2))
        target_12 = numpy.zeros((12, 12, 2))
        targets = (target_40, target_32, target_24, target_16, target_12)
        scale_ranges = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
        for count, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            h = y2 - y1
            w = x2 - x1
            c_x = (x1 + x2) / 2
            c_y = (y1 + y2) / 2
            gt_areas = (h * w) ** 0.5
            for target, scale_range in zip(targets, scale_ranges):
                if scale_range[0] < gt_areas < scale_range[1]:
                    x1, y1, x2, y2 = c_x-0.1*w, c_y-0.1*h, c_x+0.1*w, c_y+0.1*h
                    num_cell = target.shape[0]
                    cell_size = 640 / num_cell
                    i_s, i_end, j_s, j_end = x1 // cell_size, x2 // cell_size, y1 // cell_size, y2 // cell_size
                    i_s, i_end, j_s, j_end = int(i_s), int(i_end), int(j_s), int(j_end)
                    for i in range(i_s, i_end + 1):
                        for j in range(j_s, j_end + 1):
                            target[j, i] = numpy.array([1, count])
        return target_40, target_32, target_24, target_16, target_12


if __name__ == '__main__':
    from tqdm import tqdm

    dataset = MyDataset()
    l = len(dataset)
    for i in tqdm(range(l)):
        dataset[i]
