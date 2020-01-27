from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import yaml
from glob import glob
import numpy as np
import mproc
import pytorch_models.deeplab.model as dm
import pytorch_models.unet.model as um
import torch_kit.helpers as H
import image_kit.io as io
from image_kit.handler import process_input
import utils.helpers as h
from config import PRODUCTS_DIR, TILE_MAP_PATH
from config import MEANS, STDEVS
from config import DEFAULT_MODEL_TYPE, DEFAULT_NB_INPUT_CH
from config import MODEL_CONFIG_FILE, CLI_DIR



#
# CONSTANTS
#
TILE_MAP=h.read_pickle(TILE_MAP_PATH)




#
# IO
#
def image(tile_key):
    im=io.read(TILE_MAP[tile_key],return_profile=False)
    return process_input(im,means=MEANS,stdevs=STDEVS,band_indices=['ndvi'])


def batch(batch_keys):
    ims=mproc.map_with_threadpool(image,batch_keys,len(batch_keys))
    return np.stack(ims)




#
# METHODS
#
def meta(product,*keys):
    """ get product meta data
        - product: str 
        - *keys: ordered sequence of dictionary keys
    """
    meta=yaml.safe_load(open('{}/{}.yaml'.format(PRODUCTS_DIR,product)))
    for key in keys:
        meta=meta[key]
    return meta



def model(config,init_weights=None):
    """ load (eval) model from config """
    if isinstance(config,str):
        config=model_config(config)
    model_type=config.pop('type',DEFAULT_MODEL_TYPE)
    config['out_ch']=config.get('out_ch',DEFAULT_NB_INPUT_CH)
    if model_type=='dlv3p':
        model=dm.DeeplabV3plus
    elif model_type=='unet':
        model=um.UNet
    else:
        raise ValueError(f'model_type ({model_type}) not implemented')
    model=H.get_model(
        model,
        config,
        init_weights=init_weights)
    model=model.eval()
    return model


def model_config(model_name,key='models',file=MODEL_CONFIG_FILE):
    meta=yaml.safe_load(open('{}/{}.yaml'.format(CLI_DIR,MODEL_CONFIG_FILE)))
    if key: meta=meta[key]
    if model_name: meta=meta[model_name]
    return meta


def available_weights(model_name):
    return glob(f'{CLI_DIR}/weights/*{model_name}*')



