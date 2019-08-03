import os
import ipdb
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import sys



from config import cfg
from lib.net.deeplabv3plus import deeplabv3plus
from lib.net.sync_batchnorm.replicate import patch_replication_callback

DATA_RESCALE = (512, 512)
Kernel_Size = (13, 13)
img_dir = '/nfs-data/wangyf/Output/attr_seg/test_image_51'
save_dir = '/nfs-data/wangyf/Output/attr_seg/deeplabv3+/test_image_51/gray_dilate_13x13'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

def ToTensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     ## TODO
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).cuda()
    return img


def Todilate(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, Kernel_Size)
    dilate = cv2.dilate(img, kernel, iterations=1)
    return dilate


def Togray(img, dilate):
    x_index, y_index = np.where(dilate==0)
    for i in range(len(x_index)):
        x, y = x_index[i], y_index[i]
        if img[x, y, 0] == 0 and img[x, y, 1] == 0 and img[x, y, 2] == 0:
            img[x, y, 0] = 127
            img[x, y, 1] = 127
            img[x, y, 2] = 127
    return img


def output_show():
    file_list = os.listdir(img_dir)
    img_list = []
    for name in file_list:
        if name.split('.')[1] == 'jpg' or name.split('.')[1] == 'png':
            img_list.append(name)
    img_list.sort()
    net = deeplabv3plus(cfg)
    print('net initialize')
    if cfg.TEST_CKPT is None:
        raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
    device = torch.device('cuda')
    if cfg.TEST_GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    net.to(device)
    print('start loading model %s' % cfg.TEST_CKPT)
    model_dict = torch.load(cfg.TEST_CKPT, map_location=device)
    net.load_state_dict(model_dict)
    net.eval()

    for img_name in img_list:
        img_path = os.path.join(img_dir, img_name)
        img1 = cv2.imread(img_path)
        if img1.shape[0] >= 80 and img1.shape[1] >= 48:
            img2 = cv2.resize(img1, dsize=DATA_RESCALE, interpolation=cv2.INTER_CUBIC)
            img3 = ToTensor(img2)
            t1 = time.time()
            predicts = net(img3)
            t2 = time.time()
            print('precess time:', t2-t1)
            result = torch.argmax(predicts, dim=1).cpu().numpy().astype(np.uint8).transpose((1, 2, 0))
            dilate = Todilate(result)
            img_merge = img2 * dilate[:, :, np.newaxis]
            img_merge_gray = Togray(img_merge.copy(), dilate)
            # cv2.imshow('1', img_merge_gray)
            # cv2.waitKey(0)
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, img_merge_gray)
        else:
            continue

if __name__ == '__main__':
    output_show()