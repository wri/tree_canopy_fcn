import os,sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)
from importlib import reload
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import re
from pprint import pprint
import numpy as np
import pandas as pd
import utils.load as load
import utils.dataloader as dldr
from glob import glob
import torch
import mproc
import torch_kit.helpers as H
import image_kit.io as io
import image_kit.handler as hand
#
# RUN CONFIG
# 
REGION_NAME='costa_rica-san_jose'
PATHS_CSV='/home/ericp/tree_canopy_fcn/repo/datasets/costa_rica-san_jose.plieades.STATS.csv'
# HISTORY='pleiades.pleiades-green_bu-ndvi-ndwi.2020-06-04T04:56:29.p'
HISTORY='pleiades_dlv3p.pleiades-dlv3p-green_bu-ndvi-ndwi.2020-06-05T14:19:29.p'
WEIGHTS=re.sub(r'\.2020-','.best.2020-',HISTORY)
CAT_KEYS=['bu','grass','green']
CAT_BOUNDS=dldr.get_category_bounds('pleiades',*CAT_KEYS)
pprint(CAT_BOUNDS)
INPUT_PREFIX='ab_pleiades'
OUTPUT_PREFIX='ulu'
INDICES=['ndvi','ndwi']
""" plieades CR-SJ stats """
MEANS=[109.52242255601726, 100.04140528694528, 84.14917010948307, 118.72171506725374]
STDEVS=[46.6514205427311, 44.319290487368946, 45.27177332803344, 36.79198903291752]
INPUT_BANDS=[0,1,2,3]
PATH_SELECTOR=f'/DATA/imagery/{REGION_NAME}/v1/pleiades/*.tif'
BATCH_SIZE=6
MAX_PROCESSES=6
LIMIT=None
CROPPING=16
PRED_VERSION='v2'
MAX_BLACK_PIXEL=4*512

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



def create_paths_df():
    return pd.DataFrame(glob(PATH_SELECTOR),columns=['path'])


print('RUN:',MODEL_NAME)
print(datetime.now().strftime(TS_FMT))
#
# DATA
#
df=pd.read_csv(PATHS_CSV)
df=df[df.black_pixel_count<MAX_BLACK_PIXEL]
paths=df.path.tolist()[:LIMIT]
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
        pred_folder=f'{pred_folder}/{PRED_VERSION}/'
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




