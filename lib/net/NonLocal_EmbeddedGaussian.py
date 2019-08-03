#-*-coding:utf-8-*-

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

import ipdb

class NonLocal_EmbeddedGaussian(nn.Module):
    def __init__(self, dim_in, dim_inner, dim_out, max_pool_stride=2): ## TODO: in, out
        super(NonLocal_EmbeddedGaussian, self).__init__()
        self.dim_inner = dim_inner

        self.theta = nn.Conv2d(dim_in, dim_inner, 1, stride=1, padding=0) ##TODO 1
        ## phi and g: half spatial size, (N, 1024, 32, 32)->(N, 1024, 32, 32)
        self.pool = nn.MaxPool2d(kernel_size=max_pool_stride, stride=max_pool_stride, padding=0)
        self.phi = nn.Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)
        self.g = nn.Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)
        self.out = nn.Conv2d(dim_inner, dim_out, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(dim_out)
        self.apply(self._init_modules)

    def _init_modules(self, m):
        if isinstance(m, nn.Conv2d):
            init.normal_(m.weight, std=0.01)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1) ## TODO
            m.bias.data.zero_()   ## TODO

    def forward(self, x):

        batch_size = x.size(0)

        # theta_x=>(n, c, h, w)->(n, c, hw)->(n, hw, c)
        theta_x = self.theta(x).view(batch_size, self.dim_inner, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # pool_x=>(n, c, h/2, w/2)
        pool_x = self.pool(x)

        # phi_x=>(n, c, h/2, w/2)->(n, c, hw/4)
        phi_x = self.phi(pool_x).view(batch_size, self.dim_inner, -1)

        # g_x=>(n, c, h/2, w/2)->(n, c, hw/4)
        g_x = self.g(pool_x).view(batch_size, self.dim_inner, -1)

        # theta_phi=>(n, hw, c) * (n, c, hw/4)->(n, hw, hw/4)
        # ipdb.set_trace()
        theta_phi = torch.matmul(theta_x, phi_x)

        theta_phi = theta_phi * (self.dim_inner ** -.5)  ##TODO

        # p_x=>(n, hw, hw/4)->(n, hw/4, hw)
        p_x = F.softmax(theta_phi, dim=-1)
        p_x = p_x.permute(0, 2, 1)

        # t_x=>(n, c, hw/4) * (n, hw/4, hw)->(n, c, hw)->(n, c, h, w)
        t_x = torch.matmul(g_x, p_x)  ##TODO
        t_x = t_x.view(batch_size, self.dim_inner, *x.size()[2:])

        # (n, c, h, w)
        y = self.out(t_x)
        y = self.bn(y)

        return y + x


if __name__ == '__main__':
    img = torch.zeros(4, 256, 20, 20).cuda()
    net = NonLocal_EmbeddedGaussian(256, 128, 256).cuda()
    out = net(img)
    print(out.size())














