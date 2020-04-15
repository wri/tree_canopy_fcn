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
import utils.dlabs as dlabs
from config import PRODUCTS_DIR, REGIONS_DIR
from config import TILE_MAP_PATH, ALPHA_BAND
from config import MEANS, STDEVS
from config import DEFAULT_MODEL_TYPE, DEFAULT_NB_INPUT_CH
from config import MODEL_CONFIG_FILE, CLI_DIR, TILES_DIR



#
# CONSTANTS
#
# TILE_MAP=h.read_pickle(TILE_MAP_PATH)
YEAR_ERROR='treecover.load: year required for DL downloads'




#
# IO
#
# def image(tile_key):
#     im=io.read(TILE_MAP[tile_key],return_profile=False)
#     nodata=_nodata(im)
#     return process_input(im,means=MEANS,stdevs=STDEVS,band_indices=['ndvi']), nodata


def dl_image(tile_key,year,start=None,end=None,alpha_band=ALPHA_BAND):
    if not year:
        raise ValueError(YEAR_ERROR)
    im=dlabs.mosaic(tile_key,year=year,start=start,end=end,alpha_band=alpha_band)
    if im is False:
        return False, False
    else:
        if alpha_band:
            nodata=(im[4]==0)
        else:
            nodata=_nodata(im)
        if isinstance(im,np.ma.core.MaskedArray):
            msk=im.mask
            im=im.data
            im[msk]=0
            nodata=nodata | msk[0]
        im=np.nan_to_num(im)
        return process_input(im[:4],means=MEANS,stdevs=STDEVS,band_indices=['ndvi']), nodata


def batch(batch_keys):
    ims=mproc.map_with_threadpool(image,batch_keys,max_processes=len(batch_keys))
    ims,nodatas=zip(*ims)
    return np.stack(ims),np.stack(nodatas)


def dl_batch(batch_keys,year,start=None,end=None,alpha_band=ALPHA_BAND):
    if not year:
        raise ValueError(YEAR_ERROR)
    def _dl_image(tile_key):
        return dl_image(tile_key,year,start=start,end=end,alpha_band=alpha_band)
    ims=mproc.map_with_threadpool(_dl_image,batch_keys,max_processes=len(batch_keys))
    inpts=[]
    nodatas=[]
    for im,nodata in ims:
        if im is not False:
            inpts.append(im)
            nodatas.append(nodata)
    if inpts:
        return np.stack(inpts), np.stack(nodatas)
    else:
        return False, False

#
# KWARGS
#
def product(product):
    kwargs=meta(product,'product')
    kwargs['name']=kwargs.get('name',kwargs['id'].upper())
    return kwargs


def bands(product):
    _meta=meta(product)
    """ product.add_bands """
    bands_kwargs_list=[]
    for i,band in enumerate(_meta['bands']):
        b={}
        b['product_id']=_meta['product'].get('id')
        b['band_index']=i
        b['readers']=b.get('readers',_meta['product'].get('readers'))
        b.update(band)
        bands_kwargs_list.append(b)
    return bands_kwargs_list


#
# PATHS/META/SETUP/DATA
#
def study_area_path(region_name):
    return f'{REGIONS_DIR}/{region_name}.geojson'


def study_area(study_area):
    if isinstance(study_area,str):
        study_area=h.read_geojson(study_area_path(study_area))
    return study_area


def tile_keys_path(region_name,suffix=None,version=1):
    path=f'{TILES_DIR}/{region_name}'
    if suffix:
        path=f'{path}-{suffix}'
    if version:
        path=f'{path}.v{version}'
    return f'{path}.p'


def tile_keys(region_name=None,suffix=None,version=1,path=None):
    if not path:
        path=tile_keys_path(region_name,suffix=suffix,version=version)
    keys=h.read_pickle(path)
    print(f'{path}:',len(keys))
    return keys


def meta(product,*keys):
    """ get product meta data
        - product: str 
        - *keys: ordered sequence of dictionary keys
    """
    meta=yaml.safe_load(open('{}/{}.yaml'.format(PRODUCTS_DIR,product)))
    for key in keys:
        meta=meta[key]
    return meta


def model(config,file=MODEL_CONFIG_FILE,init_weights=None):
    """ load (eval) model from config """
    if isinstance(config,str):
        config=model_config(config,file=file)
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
    meta=yaml.safe_load(open('{}/{}.yaml'.format(CLI_DIR,file)))
    if key: meta=meta[key]
    if model_name: meta=meta[model_name]
    return meta


def available_weights(model_name):
    return glob(f'{CLI_DIR}/weights/*{model_name}*')



#
# INTERNAL
#
def _nodata(im):
    return (im[:3].sum(axis=0)==0)






