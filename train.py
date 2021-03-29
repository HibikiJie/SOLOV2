from models.net import Solo
from utiles.dataset import MyDataset
from utiles.loss_function import FocalLoss, DiceLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import torch
import cv2, numpy


class Trainer:

    def __init__(self):
        self.net = Solo()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(self.device)
        self.dataset = MyDataset()
        self.data_loader = DataLoader(self.dataset, 1, True, num_workers=1)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00001)

        self.f_loss = FocalLoss(alpha=0.75)
        # torch.nn.MSELoss
        self.dice_loss = FocalLoss(alpha=0.75)
        # self.net.load_state_dict(torch.load('weights/solo.pt', map_location='cpu'))

    def __call__(self):
        i = 0
        epoch = 0  # 训练轮次
        accumulation_steps = 50  # 梯度累积步数
        self.net.train()
        while True:
            loss_sum = 0
            loss1_sum = 0
            loss2_sum = 0
            loss3_sum = 0
            loss4_sum = 0
            loss5_sum = 0
            for image, maskes, target_40, target_32, target_24, target_16, target_12 in tqdm(self.data_loader):
                images = image.to(self.device)
                maskes = maskes.to(self.device)
                target_40 = target_40.to(self.device)
                target_32 = target_32.to(self.device)
                target_24 = target_24.to(self.device)
                target_16 = target_16.to(self.device)
                target_12 = target_12.to(self.device)
                # print(images.shape)
                # print(maskes.shape)
                # print(target_40.shape)

                (f_4_c, f_4_k), (f_8_c, f_8_k), (f_16_c, f_16_k), (f_32_c, f_32_k), (
                    f_64_c, f_64_k), mask_feature = self.net(images)
                # print(p80.shape)
                # print(p40.shape)
                # print(p20.shape)
                # print(mask_feature.shape)
                # exit()

                '''计算损失'''
                loss1 = self.compute_loss(f_4_c, f_4_k, target_40, mask_feature, maskes)
                loss2 = self.compute_loss(f_8_c, f_8_k, target_32, mask_feature, maskes)
                loss3 = self.compute_loss(f_16_c, f_16_k, target_24, mask_feature, maskes)
                loss4 = self.compute_loss(f_32_c, f_32_k, target_16, mask_feature, maskes)
                loss5 = self.compute_loss(f_64_c, f_64_k, target_12, mask_feature, maskes)
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                # print(loss.item())
                '''反向传播，梯度更新'''
                # self.optimizer.zero_grad()
                loss.backward()
                # self.optimizer.step()
                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                '''统计损失信息'''
                i += 1
                loss1_sum += loss1.item()
                loss2_sum += loss2.item()
                loss3_sum += loss3.item()
                loss4_sum += loss4.item()
                loss5_sum += loss5.item()
                loss_sum += loss.item()
                if (i + 1) % 1000 == 0:
                    torch.save(self.net.state_dict(), 'weights/solo.pt')
                    print(epoch,loss.item())
            epoch += 1
            '''写日志文件'''
            logs = f'''{epoch},loss_sum: {loss_sum / len(self.data_loader)},loss_64:{loss1_sum / len(self.data_loader)},loss_32:{loss2_sum / len(self.data_loader)},loss_16:{loss3_sum / len(self.data_loader)},{loss4_sum},{loss5_sum}'''
            print(logs)
            torch.save(self.net.state_dict(), 'weights/solo.pt')
            with open('logs.txt', 'a') as file:
                file.write(logs + '\n')

    def compute_loss(self, cate, kernel, target, mask_feature, maskes):
        # print(cate.shape,kernel.shape,target.shape,maskes.shape)
        # print(maskes.shape)
        positive = target[:, :, :, 0] == 1
        negative = target[:, :, :, 0] == 0
        # plt.imshow(target[:, :, 0] == 1)
        # plt.show()
        target_positive = target[positive]
        target_negative = target[negative]
        cate_positive = cate[positive]
        cate_negative = cate[negative]
        kernel_positive = kernel[positive]
        number, _ = target_positive.shape
        '''置信度损失'''
        if number > 0:
            # print(cate_positive[:, 0])
            loss_c_p = self.f_loss(cate_positive[:, 0], target_positive[:, 0].float())
            # print(loss_c_p)
            # print('positive')
        else:
            loss_c_p = 0
        loss_c_n = self.f_loss(cate_negative[:, 0], target_negative[:, 0].float())
        loss_c = loss_c_n + loss_c_p

        '''mask损失'''
        loss_mask = 0
        mask = []
        for i in range(number):
            mask.append(self.get_mask(mask_feature, kernel_positive[i]))
            # print(mask.max())
            # mask_ = (mask.reshape(160, 160) * 255).detach().cpu().numpy().astype(numpy.uint8)
            # cv2.imshow('a', mask_)
            # cv2.waitKey(1)
            # print(mask.shape,maskes[:, target_positive[i, 1].long()].shape)
        if len(mask) == 0:
            pass
        else:
            mask = torch.cat(mask, dim=1)
        if number > 0:
            loss_mask = self.dice_loss(mask, maskes[:, target_positive[:, 1].long()])
        else:
            loss_mask = 0
        # print(loss_mask)
        # print(loss_mask)
        return loss_mask + loss_c

    def get_mask(self, mask_feature, kernel):
        return torch.sigmoid(F.conv2d(mask_feature, kernel.reshape(1, 256, 1, 1)))


if __name__ == '__main__':
    trainer = Trainer()
    trainer()
