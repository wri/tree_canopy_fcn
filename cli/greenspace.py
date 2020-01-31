import os,sys
PROJECT_DIR=f'..'
sys.path.append(PROJECT_DIR)
from pprint import pprint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_kit.functional as F
from torch_kit.optimizers.radam import RAdam
import pytorch_models.deeplab.model as dm
import pytorch_models.unet.model as um
from utils.dataloader import UrbanTreeDataset
from config import GREENSPACE_THRESHOLDS


#
# RUN CONFIG
#
CROPPING=False
FLOAT_CROPPING=18
REGION='all'
RESOLUTION=1
TRAIN='train'
VALID='valid'
BATCH_SIZE=8
NB_EPOCHS=50
DEV=False
DEFAULT_OPTIMIZER='adam'
LRS=[1e-3,1e-4]
NB_CATEGORIES=2
MEANS=[101.12673535231546, 100.36417761244, 94.04471640665643, 113.85310697286442]
STDEVS=[39.0196407883084, 35.3287659336378, 33.68392945659178, 35.37087488392215]


#
# TORCH_KIT CLI
#
def model(**cfig):   
    _header('model',cfig)
    model_type=cfig.pop('type','dlv3p')
    cfig['out_ch']=cfig.get('out_ch',NB_CATEGORIES)
    if model_type=='dlv3p':
        mod=dm.DeeplabV3plus(**cfig)
    elif model_type=='unet':
        mod=um.UNet(**cfig)
    else:
        raise ValueError(f'model_type ({model_type}) not implemented')
    if torch.cuda.is_available():
        mod=mod.cuda()
    return mod


def criterion(**cfig):
    ignore_index=cfig.get('ignore_index')
    weights=cfig.get('weights')
    print("criterion:",ignore_index,weights)
    if weights:
        weights=torch.Tensor(weights)
        if torch.cuda.is_available():
            weights=weights.cuda()
    if ignore_index:
        criterion=nn.CrossEntropyLoss(weight=weights,ignore_index=ignore_index)
    else:
        criterion=nn.CrossEntropyLoss(weight=weights)
    return criterion


def optimizer(**cfig):
    _header('optimizer',cfig)
    opt_name=cfig.get('name',DEFAULT_OPTIMIZER)
    if opt_name=='adam':
        optimizer=torch.optim.Adam
    elif opt_name=='radam':
        optimizer=RAdam
    else:
        ValueError(f'optimizer "{opt_name}" not implemented')
    return optimizer


def loaders(**cfig):
    """
    """
    # INITAL DATASET HANDLING
    train_df=pd.read_csv(f'{PROJECT_DIR}/datasets/train_5percSMLA.csv')
    valid_df=pd.read_csv(f'{PROJECT_DIR}/datasets/valid_5percSMLA.csv')
    df=pd.concat([train_df,valid_df])
    #
    # on with the show
    #
    dev=cfig.get('dev')
    vmap=cfig.get('vmap')
    batch_size=cfig.get('batch_size',BATCH_SIZE)
    band_indices=['ndvi']
    augment=cfig.get('augment',True)
    shuffle=cfig.get('shuffle',True)
    update_version=cfig.get('update_version',False)

    print('AUGMENT:',augment)
    print('SHUFFLE:',shuffle)
    print('BATCH_SIZE:',batch_size)

    if (train_df.shape[0]>batch_size*8) and (valid_df.shape[0]>batch_size*2):
        if dev:
            train_df=train_df.sample(batch_size*8)
            valid_df=valid_df.sample(batch_size*2)


        dl_train=UrbanTreeDataset.loader(
            batch_size=batch_size,
            height_thresholds=GREENSPACE_THRESHOLDS,
            band_indices=band_indices,
            dataframe=train_df,
            means=MEANS,
            stdevs=STDEVS,
            augment=augment,
            train_mode=True,
            target_dtype=np.int,
            shuffle_data=shuffle)


        dl_valid=UrbanTreeDataset.loader(
            batch_size=batch_size,
            height_thresholds=GREENSPACE_THRESHOLDS,
            band_indices=band_indices,
            dataframe=valid_df,
            means=MEANS,
            stdevs=STDEVS,
            augment=augment,
            train_mode=True,
            target_dtype=np.int,
            shuffle_data=shuffle)


        print("SIZE:",train_df.shape[0],valid_df.shape[0])

        """
        dl_train=UrbanTreeDataset.loader(
            batch_size=batch_size,
            band_indices=band_indices,
            dataframe=df,
            means=MEANS,
            stdevs=STDEVS,
            augment=augment,
            value_map=vmap, 
            train_mode=True,
            target_expand_axis=None,
            UPDATE_VERSION=update_version,
            shuffle_data=shuffle)

        dl_valid=None

        print("SIZE:",df.shape[0])
        """

        return dl_train, dl_valid
    else:
        print('NOT ENOUGH DATA',train_df.shape[0],valid_df.shape[0],batch_size*8,batch_size*30)
        return False, False

#
# HELPERS
#
def _header(title,cfig=None):
    print('='*100)
    print(title)
    print('-'*100)    
    if cfig:
        pprint(cfig)


