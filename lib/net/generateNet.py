# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from lib.net.deeplabv3plus import deeplabv3plus
# from lib.net.deeplabv3plus_nonlocal import deeplabv3plus_nonlocal
# from net.supernet import SuperNet
# from net.EANet import EANet
# from net.DANet import DANet
# from net.deeplabv3plushd import deeplabv3plushd
# from net.DANethd import DANethd
def generate_net(cfg):
	if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
		return deeplabv3plus(cfg)
		#return deeplabv3plus_nonlocal(cfg)

	# if cfg.MODEL_NAME == 'supernet' or cfg.MODEL_NAME == 'SuperNet':
	# 	return SuperNet(cfg)
	# if cfg.MODEL_NAME == 'eanet' or cfg.MODEL_NAME == 'EANet':
	# 	return EANet(cfg)
	# if cfg.MODEL_NAME == 'danet' or cfg.MODEL_NAME == 'DANet':
	# 	return DANet(cfg)
	# if cfg.MODEL_NAME == 'deeplabv3plushd' or cfg.MODEL_NAME == 'deeplabv3+hd':
	# 	return deeplabv3plushd(cfg)
	# if cfg.MODEL_NAME == 'danethd' or cfg.MODEL_NAME == 'DANethd':
	# 	return DANethd(cfg)
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
