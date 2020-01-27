from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import yaml
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



#
# CONSTANTS
#
TILE_MAP=h.read_pickle(TILE_MAP_PATH)




#
# IO
#
def read_image(tile_key):
    im=io.read(TILE_MAP[tile_key],return_profile=False)
    return process_input(im,means=MEANS,stdevs=STDEVS,band_indices=['ndvi'])


def read_image_batch(batch_keys):
    ims=mproc.map_with_threadpool(read_image,batch_keys,len(batch_keys))
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



def model(model_config,init_weights=None):
    """ load (eval) model from config """
    model_type=model_config.pop('type',DEFAULT_MODEL_TYPE)
    model_config['out_ch']=model_config.get('out_ch',DEFAULT_NB_INPUT_CH)
    if model_type=='dlv3p':
        model=dm.DeeplabV3plus
    elif model_type=='unet':
        model=um.UNet
    else:
        raise ValueError(f'model_type ({model_type}) not implemented')
    model=H.get_model(
        model,
        model_config,
        init_weights=init_weights)
    model=model.eval()
    return mod


