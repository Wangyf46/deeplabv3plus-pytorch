# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from lib.datasets.VOCDataset import VOCDataset
from lib.datasets.V1datasets import V1datasets

def generate_dataset(dataset_name, cfg, period, aug=False):
	if dataset_name == 'voc2012' or dataset_name == 'VOC2012':
		return VOCDataset('VOC2012', cfg, period, aug)
	elif dataset_name == 'clean_datasets_V1':
		return V1datasets('clean_datasets_V1', cfg, period, aug)
	else:
		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)

