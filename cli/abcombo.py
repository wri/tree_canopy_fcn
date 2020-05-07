import os,sys
PROJECT_DIR=f'..'
sys.path.append(PROJECT_DIR)
from pprint import pprint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_kit.loss import MaskedLoss
import torch_kit.functional as F
from torch_kit.optimizers.radam import RAdam
import pytorch_models.deeplab.model as dm
import pytorch_models.unet.model as um
from utils.dataloader import HeightIndexDataset, CATEGORY_BOUNDS
from config import BUILTUP_CATEGORY_THRESHOLDS


#
# RUN CONFIG
#
BATCH_SIZE=8
DEFAULT_OPTIMIZER='adam'
LRS=[1e-3,1e-4]
NB_CATEGORIES=len(CATEGORY_BOUNDS)+1
# # STATS: ALL
# MEANS=[100.83741572079242, 100.4938850966076, 86.63500986931308, 118.72746674454453]
# STDEVS=[42.098045003124774, 39.07388735786421, 39.629813116928815, 34.72351480486876]
# STATS: 2015,16 Train/valid
MEANS=[94.79936157686979, 92.8912348691044, 80.50194782393349, 108.14889758142212]
STDEVS=[36.37876660224377, 33.22686387734999, 33.30808192430284, 30.075380846943716]
DSETS_PATH='../datasets/los_angeles-plieades-lidar_USGS_LPC_CA_LosAngeles_2016_LAS_2018.STATS.csv'
YEAR_MAX=2016
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
    if ignore_index is not None:
        # criterion=nn.CrossEntropyLoss(weight=weights,ignore_index=ignore_index)
        criterion=MaskedLoss(
            weight=weights,
            loss_type='ce',
            mask_value=ignore_index )
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
    dsets_df=pd.read_csv(DSETS_PATH)
    train_df=dsets_df[dsets_df.dset_type=='train']
    valid_df=dsets_df[dsets_df.dset_type=='valid']
    train_df=train_df[train_df.year<=YEAR_MAX]
    valid_df=valid_df[valid_df.year<=YEAR_MAX]
    example_path=train_df.rgbn_path.iloc[0]
    #
    # on with the show
    #
    dev=cfig.get('dev')
    vmap=cfig.get('vmap')
    batch_size=cfig.get('batch_size',BATCH_SIZE)
    band_indices=['ndvi']
    augment=cfig.get('augment',True)
    shuffle=cfig.get('shuffle',True)
    no_data_value=cfig.get('no_data_value',False)
    cropping=cfig.get('cropping',None)
    float_cropping=cfig.get('float_cropping',None)
    update_version=cfig.get('update_version',False)

    print('AUGMENT:',augment)
    print('SHUFFLE:',shuffle)
    print('BATCH_SIZE:',batch_size)
    print('NO DATA VALUE:',no_data_value)
    print('CROPPING:',cropping)
    print('FLOAT CROPPING:',float_cropping)

    if (train_df.shape[0]>batch_size*8) and (valid_df.shape[0]>batch_size*2):
        if dev:
            train_df=train_df.sample(batch_size*8)
            valid_df=valid_df.sample(batch_size*2)


        dl_train=HeightIndexDataset.loader(
            batch_size=batch_size,
            band_indices=['ndvi','ndwi'],
            dataframe=train_df,
            means=MEANS,
            stdevs=STDEVS,
            no_data_value=no_data_value,
            cropping=cropping,
            float_cropping=float_cropping,
            example_path=example_path,
            augment=augment,
            train_mode=True,
            target_dtype=np.int,
            shuffle_data=shuffle)


        dl_valid=HeightIndexDataset.loader(
            batch_size=batch_size,
            band_indices=['ndvi','ndwi'],
            dataframe=valid_df,
            means=MEANS,
            stdevs=STDEVS,
            no_data_value=no_data_value,
            cropping=cropping,
            float_cropping=float_cropping,
            example_path=example_path,
            augment=augment,
            train_mode=True,
            target_dtype=np.int,
            shuffle_data=shuffle)


        print("SIZE:",train_df.shape[0],valid_df.shape[0])

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


