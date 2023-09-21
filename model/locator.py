from model.HR_Net.seg_hrnet import get_seg_model

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import models
import cv2


class Crowd_locator(nn.Module):
    def __init__(self, net_name, gpu_id, pretrained=True):
        super(Crowd_locator, self).__init__()

        if net_name == 'dclnet':
            self.Extractor = get_seg_model()

        if len(gpu_id) > 1:
            self.Extractor = nn.DataParallel(self.Extractor).cuda()
        else:
            self.Extractor = self.Extractor.cuda()

        self.loss_BCE = nn.BCELoss().cuda()

    @property
    def loss(self):
        return  self.head_map_loss, self.binar_map_loss

    def forward(self, img, mask_gt, mode = 'train'):
        # print(size_map_gt.max())
        pre_map = self.Extractor(img)

        binar_map = self.BinarizedF(pre_map,threshold=0.5)


        if mode == 'train':
        # weight = torch.ones_like(binar_map).cuda()
        # weight[mask_gt==1] = 2
            assert pre_map.size(2) == mask_gt.size(2)
            self.binar_map_loss = (torch.abs(binar_map-mask_gt)).mean()

            self.head_map_loss = F.mse_loss(pre_map, mask_gt)

        return pre_map,binar_map

    def test_forward(self, img):
        pre_map = self.Extractor(img)

        return pre_map
    
    def BinarizedF(self, input, threshold):
        a = torch.ones_like(input).cuda()
        b = torch.zeros_like(input).cuda()
        output = torch.where(input>=threshold,a,b)
        return output


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    import torch.nn.functional as F
   
