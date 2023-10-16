"""Graph matching config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
from easydict import EasyDict as edict
import numpy as np

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C



# HHGM model options
__C.HHGM = edict()
__C.HHGM.FEATURE_NODE_CHANNEL = 1024  #节点的特征通道
__C.HHGM.FEATURE_EDGE_CHANNEL = 1024  #边的特征通道
__C.HHGM.BS_ITER_NUM = 20    #BS双随机化参数---最大迭代次数
__C.HHGM.BS_EPSILON = 1.0e-10    #BS双随机化参数---最小的eps
__C.HHGM.VOTING_ALPHA = 200. #投票层参数，增大差异
__C.HHGM.GNN_LAYER = 5   #嵌入层的gnn层数
__C.HHGM.GNN_FEAT = 1024*2     #嵌入层的gnn输出特征通道数
__C.HHGM.POINTER = ''     #transformer
__C.HHGM.SKADDCR = False     #对于sk算法是否需要增加新的行和列
__C.HHGM.SKADDCRVALUE = 0.0    #对于sk算法是否需要增加新的行和列的值
__C.HHGM.USEINLIERRATE = False
__C.HHGM.NORMALS = False
__C.HHGM.FEATURES = ['xyz']
__C.HHGM.NEIGHBORSNUM = 20
__C.HHGM.USEATTEND = 'attentiontransformer'



# Transformer
__C.attention_type= 'dot_prod'
__C.nhead= 8
__C.d_embed= 2048
__C.d_feedforward= 1024
__C.dropout= 0.0
__C.pre_norm= True
__C.transformer_act= 'relu'

# Transformer encoder
__C.num_encoder_layers= 6
__C.transformer_encoder_has_pos_emb= True
__C.sa_val_has_pos_emb= True
__C.ca_val_has_pos_emb= True
__C.pos_emb_type= 'learned'  # either 'sine' or 'learned'



