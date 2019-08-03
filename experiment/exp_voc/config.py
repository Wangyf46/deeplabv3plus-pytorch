# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time

class Configuration():
    def __init__(self):
        self.DATE = time.strftime("%Y-%m-%d", time.localtime())
        self.EXP_NAME = 'exp_V1'
        self.ROOT_DIR = '/nfs-data/wangyf/datasets/'

        self.DATA_NAME = 'clean_datasets_V1'
        self.DATA_AUG = False
        self.DATA_WORKERS = 8
        self.DATA_RESCALE = 512
        self.DATA_RANDOMCROP = 512
        self.DATA_RANDOMROTATION = 0
        self.DATA_RANDOMSCALE = 2
        self.DATA_RANDOM_H = 10
        self.DATA_RANDOM_S = 10
        self.DATA_RANDOM_V = 10
        self.DATA_RANDOMFLIP = 0.5

        self.MODEL_NAME = 'deeplabv3plus'
        self.MODEL_BACKBONE = 'res101_atrous'
        #self.MODEL_BACKBONE = 'xception'

        self.MODEL_OUTPUT_STRIDE = 16
        self.MODEL_ASPP_OUTDIM = 256
        self.MODEL_SHORTCUT_DIM = 48
        self.MODEL_SHORTCUT_KERNEL = 1
        self.MODEL_NUM_CLASSES = 2 #
        self.MODEL_SAVE_DIR = os.path.join('../../model', self.EXP_NAME, self.DATE, self.MODEL_BACKBONE)

        self.TRAIN_LR = 0.007
        self.TRAIN_LR_GAMMA = 0.1
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_WEIGHT_DECAY = 0.00004
        self.TRAIN_BN_MOM = 0.0003
        self.TRAIN_POWER = 0.9
        self.TRAIN_GPUS = 10
        self.TRAIN_BATCHES = 60
        self.TRAIN_SHUFFLE = True
        self.TRAIN_MINEPOCH = 0
        self.TRAIN_EPOCHS = 100
        self.TRAIN_LOSS_LAMBDA = 0
        self.TRAIN_TBLOG = True
        # self.TRAIN_CKPT = os.path.join('../../model/exp_V1/2019-07-03/res101_atrous/deeplabv3plus_res101_atrous_clean_datasets_V1_itr42639.pth')

        self.LOG_DIR = os.path.join('../../log', self.EXP_NAME, self.DATE, self.MODEL_BACKBONE)
        self.Log_File = self.LOG_DIR + '/record.txt'

        self.TEST_MULTISCALE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        self.TEST_FLIP = True
        # self.TEST_CKPT = os.path.join('../../model/exp_voc/2019-04-22/xception/deeplabv3plus_xception_VOC2012_itr27101.pth') ##TODO
        # self.TEST_CKPT = os.path.join('../../model/exp_V1/2019-06-26/res101_atrous/deeplabv3plus_res101_atrous_clean_datasets_V1_itr93010.pth')
        self.TEST_CKPT = os.path.join(
            '../../model/exp_V1/2019-07-05/res101_atrous/deeplabv3plus_res101_atrous_clean_datasets_V1_itr67104.pth')
        self.TEST_GPUS = 4
        self.TEST_BATCHES = 32

        self.__check()
        self.__add_path('/nfs-data/wangyf/AIIT/Semantic-segmentation/deeplabv3plus-pytorch')

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not avalable')
        if self.TRAIN_GPUS == 0:
            raise ValueError('config.py: the number of GPU is 0')
        if not os.path.isdir(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        if not os.path.isdir(self.MODEL_SAVE_DIR):
            os.makedirs(self.MODEL_SAVE_DIR)

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)


cfg = Configuration() 	
