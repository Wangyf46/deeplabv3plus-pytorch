# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import sys

# sys.path.insert(0, '/nfs-data/wangyf/AIIT/Semantic-segmentation/deeplabv3plus-pytorch/')
import ipdb

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib.net.sync_batchnorm import SynchronizedBatchNorm2d
from lib.net.NonLocal_EmbeddedGaussian import NonLocal_EmbeddedGaussian


class ASPP(nn.Module):

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        #		self.conv_cat = nn.Sequential(
        #				nn.Conv2d(dim_out*4, dim_out, 1, 1, padding=0),
        #				SynchronizedBatchNorm2d(dim_out),
        #				nn.ReLU(inplace=True),
        #		)

        self.non_local = NonLocal_EmbeddedGaussian(dim_out, int(dim_out * 0.5), dim_out)  ##TODO

    def forward(self, x):
        # ipdb.set_trace()
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)    #(n, 256, row, col)
        conv3x3_1 = self.branch2(x)  #(n, 256, row, col)
        conv3x3_2 = self.branch3(x)  #(n, 256, row, col)
        conv3x3_3 = self.branch4(x)  #(n, 256, row, col)

        # global_feature = torch.mean(x,2,True)
        # global_feature = torch.mean(global_feature,3,True)

        global_feature = self.branch5(x)  #(n, 256, 7, 7)
        global_feature = F.interpolate(global_feature, (row, col))
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)  ##
        #		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3], dim=1)
        aspp_result = self.conv_cat(feature_cat)  #(n, 256, col, row)
        result = self.non_local(aspp_result) #(n, 256, col, row)
        return result


if __name__ == '__main__':
    img = torch.zeros(4, 256, 20, 20)
    net = ASPP(256, 256)  ##in==??, out=256
    out = net(img)
    print(out.size())
