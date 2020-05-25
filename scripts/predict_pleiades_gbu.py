import os,sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)
from importlib import reload
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import re
import numpy as np
import pandas as pd
import utils.load as load
from utils.dataloader import HeightIndexDataset
from glob import glob
import torch
import mproc
import torch_kit.helpers as H
import image_kit.io as io
import image_kit.handler as hand
#
# RUN CONFIG
# 
PRED_VERSION='gbu-pleiades-v1'
WEIGHTS='green_bu_plieades.green_bu_plieades-shallow.best.2020-05-15T05:38:03.p'
CAT_BOUNDS=HeightIndexDataset.NAIP_ALL
INPUT_PREFIX='ab_pleiades'
OUTPUT_PREFIX='gbu'
INDICES=['ndvi','ndwi']
""" plieades BH stats """
MEANS=[112.27539083920642, 106.95122589603547, 89.21136361353429, 112.65504157044697]
STDEVS=[40.368414410492015, 35.257702825645666, 37.21043313940668, 36.97152317592175]
""" plieades LA stats
MEANS=[94.79936157686979, 92.8912348691044, 80.50194782393349, 108.14889758142212]
STDEVS=[36.37876660224377, 33.22686387734999, 33.30808192430284, 30.075380846943716]
"""
IBNDS=None
INPUT_BANDS=None
PATH_SELECTOR='/DATA/imagery/belohorizonte/v1/pleiades/*.tif'
BATCH_SIZE=6
MAX_PROCESSES=6
LIMIT=None
CROPPING=16
#
# CONSTANTS
#
_parts=WEIGHTS.split('.')
MODEL_PYFILE=_parts[0]
MODEL_NAME=_parts[1]
AUGMENT=False
SHUF=False
FCRP=None
TAIL_REGEX='_20[0-9]{2}-(train|valid|test)'
TS_FMT="[%Y-%m-%d] %H:%M:%S"



print('RUN:',MODEL_NAME)
print(datetime.now().strftime(TS_FMT))
#
# DATA
#
paths=glob(PATH_SELECTOR)[:LIMIT]
print('NB_IMAGES:',len(paths))



#
# MODEL
# 
DEVICE=H.get_device()
model=load.model(MODEL_NAME,file=MODEL_PYFILE,init_weights=f'{PROJECT_DIR}/cli/weights/{WEIGHTS}')
model=model.eval().to(DEVICE)



#
# METHODS
#
def get_input(path):
    im,p=io.read(path,return_profile=True)
    im=hand.process_input(
        im,
        rotate=False,
        flip=False,     
        input_bands=INPUT_BANDS,
        band_indices=INDICES,
        indices_dict=None,
        padding=None,
        padding_value=0,
        cropping=CROPPING,
        bounds=IBNDS,
        means=MEANS,
        stdevs=STDEVS,
        preprocess=None)
    return im, p


def predict(im):
    pred=model(torch.Tensor(np.expand_dims(im,axis=0)).to(DEVICE))
    return pred.argmax(axis=1)

    
def prediction_path(path):
    path=re.sub('v[0-9]/','',path)
    pred_folder='/predictions/'
    if PRED_VERSION:
        pred_folder=f'{pred_folder}{PRED_VERSION}/'
    path=re.sub('/imagery/',pred_folder,path)
    path=re.sub(f'/{INPUT_PREFIX}_',f'/{OUTPUT_PREFIX}_',path)
    path=re.sub(TAIL_REGEX,'',path)
    return path


def run_prediction(path):
    inpt,p=get_input(path)
    pred=predict(inpt)
    pred=H.to_numpy(pred).astype(np.uint8)
    path=prediction_path(path)
    p['count']=1
    p=io.update_profile(p,crop=CROPPING)
    io.write(pred,path,p)
    return path



#
# RUN
#
pred_paths=mproc.map_sequential(run_prediction,paths,max_processes=MAX_PROCESSES)
print('COMPLETE:',len(pred_paths))
print(pred_paths[:2])
print(datetime.now().strftime(TS_FMT))




