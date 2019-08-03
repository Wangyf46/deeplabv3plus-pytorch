#-*-coding:utf-8-*-
# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division
import os
import sys
import pandas as pd
import cv2
import tqdm
import multiprocessing
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from lib.datasets.transform import *
import ipdb


class V1datasets(Dataset):
    def __init__(self, dataset_name, cfg, period, aug):
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(cfg.ROOT_DIR, dataset_name)
        self.rst_dir = os.path.join('../../results', cfg.EXP_NAME, cfg.DATE, cfg.MODEL_BACKBONE)
        self.eval_dir = os.path.join('../../eval_result', cfg.EXP_NAME, cfg.DATE, cfg.MODEL_BACKBONE)
        if not os.path.isdir(self.rst_dir):
            os.makedirs(self.rst_dir)
        if not os.path.isdir(self.eval_dir):
            os.makedirs(self.eval_dir)
        self.period = period
        if period == 'train':
            self.img_dir = os.path.join(self.dataset_dir, 'train_img')
            self.seg_dir = os.path.join(self.dataset_dir, 'train_gt')
            self.name_list = os.listdir(self.seg_dir)
        elif period == 'val':
            self.img_dir = os.path.join(self.dataset_dir, 'val_img')
            self.seg_dir = os.path.join(self.dataset_dir, 'val_gt')
            self.name_list = os.listdir(self.seg_dir)  ###
        else:
            self.img_dir = os.path.join('/nfs-data/wangyf/Output/seg_out/test_image_51')
            self.name_list = os.listdir(self.img_dir)
        self.rescale = None
        self.centerlize = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.multiscale = None
        self.totensor = ToTensor()
        self.cfg = cfg

        if dataset_name == 'clean_datasets_V1':
            self.categories = [
                'human']  # 1

            self.num_categories = len(self.categories)
            assert (self.num_categories+1  == self.cfg.MODEL_NUM_CLASSES)
            self.cmap = self.__colormap(len(self.categories)+1)

        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale(cfg.DATA_RESCALE, fix=False)
        if 'train' in self.period:
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        else:
            self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        img_file = self.name_list[idx]

        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.array(io.imread(img_file),dtype=np.uint8)
        r, c, _ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        if 'train' in self.period:
            seg_file = self.seg_dir + '/' + name
            segmentation = np.array(Image.open(seg_file).convert('L'))
            sample['segmentation'] = segmentation

            if self.cfg.DATA_RANDOM_H > 0 or self.cfg.DATA_RANDOM_S > 0 or self.cfg.DATA_RANDOM_V > 0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomscale(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)
            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)
        else:
            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)
            sample = self.multiscale(sample)

        if 'segmentation' in sample.keys():
            sample['mask'] = sample['segmentation'] < self.cfg.MODEL_NUM_CLASSES
            t = sample['segmentation']
            t[t >= self.cfg.MODEL_NUM_CLASSES] = 0
            sample['segmentation_onehot'] = onehot(t, self.cfg.MODEL_NUM_CLASSES)
        sample = self.totensor(sample)

        return sample

    def __colormap(self, N):
        """Get the map from label index to color

        Args:
            N: number of class

            return: a Nx3 matrix

        """
        cmap = np.zeros((N, 3), dtype=np.uint8)

        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        for i in range(N):
            r = 0
            g = 0
            b = 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap

    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r, c = m.shape
        cmap = np.zeros((r, c, 3), dtype=np.uint8)
        cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3
        cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2
        cmap[:, :, 2] = (m & 4) << 5
        return cmap

    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        i = 1
        folder_path = os.path.join(self.rst_dir, '%s_%s_cls' % (model_id, self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s' % sample['name'])
            # predict_color = self.label2colormap(sample['predict'])
            # p = self.__coco2voc(sample['predict'])
            cv2.imwrite(file_path, sample['predict'])
            print('[%d/%d] %s saved' % (i, len(result_list), file_path))
            i += 1


    def do_python_eval(self, model_id):
        predict_folder = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
        # predict_folder = '../../results/exp_V1/2019-06-29/res101_atrous/deeplabv3plus_val_cls/'
        # predict_folder = '/nfs-data/wangyf/datasets/clean_datasets_V1/val_out/'
        gt_folder = self.seg_dir
        TP = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
        P = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
        T = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
        for idx in range(len(self.name_list)):
            print('%d/%d'%(idx,len(self.name_list)))
            name = self.name_list[idx]
            '''
            str1 = name.split('.')[0]
            str2 = str1.split('-')
            if len(str2) != 1:
                if str2[-1] == 'profile':
                    str1 = str2[0]
            predict_file = predict_folder + '/' + str1 + '.png'
            if os.path.isfile(predict_file) == False:
                predict_file = predict_folder + '/' + str1 + '.jpg'
            gt_file = os.path.join(gt_folder, '%s'%name)
            predict = np.array(Image.open(predict_file))
            gt = np.array(Image.open(gt_file).convert('L').resize((96, 160), Image.ANTIALIAS))
            '''
            predict_file = os.path.join(predict_folder, '%s'%name)
            gt_file = os.path.join(gt_folder, '%s'%name)
            predict = np.array(Image.open(predict_file)) ##
            gt = np.array(Image.open(gt_file).convert('L'))
            # predict = cv2.imread(predict_file)
            # gt = cv2.imread(gt_file)
            cal = gt < 255
            mask = (predict==gt) & cal
            for i in range(self.cfg.MODEL_NUM_CLASSES):
                P[i] += np.sum((predict==i) * cal)
                T[i] += np.sum((gt==i) * cal)
                TP[i] += np.sum((gt==i) * mask)
        TP = TP.astype(np.float64)
        T = T.astype(np.float64)
        P = P.astype(np.float64)
        IoU = TP/(T+P-TP)
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            if i == 0:
                print('%15s:%7.3f%%'%('backbound', IoU[i] * 100))
            else:
                print('%15s:%7.3f%%'%(self.categories[i-1], IoU[i] * 100))
        miou = np.mean(IoU)
        print('==================================')
        print('%15s:%7.3f%%'%('mIoU',miou * 100))



