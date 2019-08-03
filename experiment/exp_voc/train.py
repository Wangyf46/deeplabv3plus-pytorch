# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import time
import sys
import numpy as np
from termcolor import cprint
import  shutil
from config import cfg
from lib.datasets.generateData import generate_dataset
from lib.net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from lib.net.loss import MaskCrossEntropyLoss, MaskBCELoss, MaskBCEWithLogitsLoss
from lib.net.sync_batchnorm.replicate import patch_replication_callback


def log_print(text, color = None, on_color = None, attrs = None, log_file = open(cfg.Log_File, 'w')):
    print(text, )
    if cprint is not None:
        cprint(text, color = color, on_color = on_color, attrs = attrs)
    else:
        print(text)


def train_net():
    dataset = generate_dataset(cfg.DATA_NAME, cfg, 'train', cfg.DATA_AUG)
    dataloader = DataLoader(dataset,
                batch_size=cfg.TRAIN_BATCHES,
                shuffle=cfg.TRAIN_SHUFFLE,
                num_workers=cfg.DATA_WORKERS,
                drop_last=True)

    net = generate_net(cfg)
    if cfg.TRAIN_TBLOG:
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(cfg.LOG_DIR)

    print('Use %d GPU'%cfg.TRAIN_GPUS)
    if cfg.TRAIN_GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    net.cuda()

    if cfg.TRAIN_CKPT:
        pretrained_dict = torch.load(cfg.TRAIN_CKPT)
        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        # net.load_state_dict(torch.load(cfg.TRAIN_CKPT),False)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.SGD(
        params = [
            {'params': get_params(net.module,key='1x'), 'lr': cfg.TRAIN_LR},
            {'params': get_params(net.module,key='10x'), 'lr': 10*cfg.TRAIN_LR}
        ],
        momentum=cfg.TRAIN_MOMENTUM
    )
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN_LR_MST, gamma=cfg.TRAIN_LR_GAMMA, last_epoch=-1)
    itr = cfg.TRAIN_MINEPOCH * len(dataloader)
    max_itr = cfg.TRAIN_EPOCHS * len(dataloader)
    print(itr, max_itr, len(dataloader))
    tblogger = SummaryWriter(cfg.LOG_DIR)
    net.train()
    for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS + 1):
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for i_batch, sample_batched in enumerate(dataloader):
            data_time.update(time.time() - end)  # measure batch_size data loading time
            now_lr = adjust_lr(optimizer, itr, max_itr)
            inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']

            labels_batched = labels_batched.long().cuda()
            #0foreground_pix = (torch.sum(labels_batched!=0).float()+1)/(cfg.DATA_RESCALE**2*cfg.TRAIN_BATCHES)
            predicts_batched = net(inputs_batched)
            predicts_batched = predicts_batched.cuda()

            loss = criterion(predicts_batched, labels_batched)
            losses.update(loss.item(), cfg.TRAIN_BATCHES)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if cfg.TRAIN_TBLOG and itr % 10 == 0:
                #inputs = np.array((inputs_batched[0]*128+128).numpy().transpose((1,2,0)),dtype=np.uint8)
                #inputs = inputs_batched.numpy()[0]
                inputs = inputs_batched.numpy()[0]/2.0 + 0.5
                labels = labels_batched[0].cpu().numpy()
                labels_color = dataset.label2colormap(labels).transpose((2,0,1))
                predicts = torch.argmax(predicts_batched[0],dim=0).cpu().numpy()
                predicts_color = dataset.label2colormap(predicts).transpose((2,0,1))
                pix_acc = np.sum(labels==predicts)/(cfg.DATA_RESCALE**2)

                print_str = 'Epoch: [{0}/{1}]\t' \
                            .format(epoch, cfg.TRAIN_EPOCHS)
                print_str += 'Batch: [{0}]/{1}\t' \
                            .format(i_batch + 1, dataset.__len__()//cfg.TRAIN_BATCHES)
                print_str += 'LR: {0}\t' \
                            .format(now_lr)
                print_str += 'Data time {data_time.cur:.3f}({data_time.avg:.3f})\t' \
                    .format(data_time=data_time)
                print_str += 'Batch time {batch_time.cur:.3f}({batch_time.avg:.3f})\t' \
                    .format(batch_time=batch_time)
                print_str += 'Loss {loss.cur:.4f}({loss.avg:.4f})\t'\
                    .format(loss=losses)
                log_print(print_str, color="green", attrs=["bold"])

                tblogger.add_scalar('loss', losses.avg, itr)
                tblogger.add_scalar('lr', now_lr, itr)
                tblogger.add_scalar('pixel acc', pix_acc, itr)
                tblogger.add_image('Input', inputs, itr)
                tblogger.add_image('Label', labels_color, itr)
                tblogger.add_image('Output', predicts_color, itr)
            end = time.time()
            itr += 1
        if epoch % 5 == 0:
            save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_itr%d.pth'%(cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, itr))
            torch.save(net.state_dict(), save_path)
    save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d_last.pth'%(cfg.MODEL_NAME, cfg.MODEL_BACKBONE, cfg.DATA_NAME, cfg.TRAIN_EPOCHS))
    torch.save(net.state_dict(), save_path)
    if cfg.TRAIN_TBLOG:
        tblogger.close()
    print('%s has been saved'%save_path)

def adjust_lr(optimizer, itr, max_itr):
    now_lr = cfg.TRAIN_LR * ((1 - float(itr)/max_itr) ** cfg.TRAIN_POWER)
    optimizer.param_groups[0]['lr'] = now_lr
    optimizer.param_groups[1]['lr'] = 10*now_lr
    return now_lr

def get_params(model, key):
    for m in model.named_modules():
        if key == '1x':
            if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur, n=1):
        self.cur = cur
        self.sum += cur * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    train_net()


