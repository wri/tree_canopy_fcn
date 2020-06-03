import os,sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
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


#
# OVERWRITE CONFIG
#
MEANS=None
STDEVS=None
DSETS_PATH=None
YEAR_MAX=None
INPUT_BANDS=None
IBNDS=None
CAT_BOUNDS=None
INDICES=None
TARGET_RGBN=None
TARGET_RGBN_AS_INPUT=None


#
# SET CONFIG
#
HARD_DEV=False
BATCH_SIZE=8
DEFAULT_OPTIMIZER='adam'
LRS=[1e-3,1e-4]
NB_CATEGORIES=len(CATEGORY_BOUNDS)+1


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
    _require([
        MEANS,
        STDEVS,
        DSETS_PATH,
        TARGET_RGBN,
        TARGET_RGBN_AS_INPUT
    ])
    # INITAL DATASET HANDLING
    dsets_df=pd.read_csv(DSETS_PATH)
    train_df=dsets_df[dsets_df.dset_type=='train']
    valid_df=dsets_df[dsets_df.dset_type=='valid']
    if YEAR_MAX:
        train_df=train_df[train_df.input_year<=YEAR_MAX]
        valid_df=valid_df[valid_df.input_year<=YEAR_MAX]
    if TARGET_RGBN_AS_INPUT:
        train_df['input_path']=train_df.rgbn_path.copy()
        valid_df['input_path']=valid_df.rgbn_path.copy()
    example_path=train_df.input_path.iloc[0]
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


    if HARD_DEV:
        train_df=train_df.iloc[1:(batch_size*8)+1]
        valid_df=valid_df.iloc[1:(batch_size*2)+1]

    print('AUGMENT:',augment)
    print('SHUFFLE:',shuffle)
    print('BATCH_SIZE:',batch_size)
    print('NO DATA VALUE:',no_data_value)
    print('CROPPING:',cropping)
    print('FLOAT CROPPING:',float_cropping)

    if (train_df.shape[0]>=batch_size*8) and (valid_df.shape[0]>=batch_size*2):
        if dev:
            train_df=train_df.sample(batch_size*8)
            valid_df=valid_df.sample(batch_size*2)

        input_bands=INPUT_BANDS or [0,1,2,3]

        dl_train=HeightIndexDataset.loader(
            batch_size=batch_size,
            dataframe=train_df,
            input_bands=input_bands,
            input_band_count=len(input_bands),
            band_indices=INDICES,
            category_bounds=CAT_BOUNDS,
            input_bounds=IBNDS,
            target_rgbn=TARGET_RGBN,
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
            dataframe=valid_df,
            input_bands=input_bands,
            input_band_count=len(input_bands),
            band_indices=INDICES,
            category_bounds=CAT_BOUNDS,
            input_bounds=IBNDS,
            target_rgbn=TARGET_RGBN,
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


def _require(required):
    for r in required:
        if r is None:
            raise ValueError('required constants are not set')






