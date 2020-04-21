#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)


# In[2]:


from pprint import pprint
import pandas as pd
import utils.helpers as h
import utils.load as load
import utils.dlabs as dlabs
import mproc


# In[3]:


YEAR_START=2015
YEAR_END=2018
PRODUCTS=['airbus:oneatlas:phr:v2']
PRODUCT_NAME='ab_pleiades'
INPUT_BANDS=['red', 'green', 'blue', 'nir', 'alpha']
ALPHA_BAND=False
DSET_TYPES=['train','valid','test']
DATA_ROOT='/DATA/imagery'
DATA_DIR='ab_test'
PREFIX=PRODUCT_NAME
YEAR_RANGE=range(YEAR_START,YEAR_END+1)
MAX_PROCESSES=8


# ---

# In[4]:


def download_tile(tile_key,dset_type):
    out=None
    for year in YEAR_RANGE:
        out=download_tile_year(tile_key,year,dset_type)
        if out: break;
    if not out:
        out={'tile_key': tile_key, 'year': None, 'path': None, 'error': False, 'error_msg': None}
    return out


def download_tile_year(tile_key,year,dset_type):
    dest=f'{DATA_DIR}/{PREFIX}_{tile_key}_{year}-{dset_type}.tif'
    out=None
    error=False
    error_msg=None
    try:
        out=dlabs.mosaic(        
            tile_key,
            products=PRODUCTS,
            bands=INPUT_BANDS,
            alpha_band=ALPHA_BAND,
            start=f'{year}-01-01',
            end=f'{year+1}-01-01',
            dest=dest)
    except Exception as e:
        error=True
        error_msg=str(e)
    if out or error:
        return {
            'tile_key': tile_key,
            'year': year,
            'path': dest,
            'error': error,
            'error_msg': error_msg}


# ---

# In[5]:


region='los_angeles'
version=1
lim=3


# In[6]:


def download_data(region,version,lim):
    data_dir=f'{DATA_ROOT}/{region}/v{version}/{PRODUCT_NAME}'
    print('DIRECTORY:',data_dir)
    h.ensure_dir(directory=data_dir)
    aoi_config=load.aoi(region)
    print('AOI:')
    pprint(aoi_config)
    for typ in DSET_TYPES:
        keys=load.tile_keys(region,typ,frac=aoi_config['sample_frac'])[:lim]
        print(f'DOWLOADING {typ}({len(keys)}):')
        def _download(key):
            return download_tile(key,typ)
        out=mproc.map_with_threadpool(_download,keys,max_processes=MAX_PROCESSES)
        df=pd.DataFrame(out)
        log_path=f'{PRODUCT_NAME}-{typ}_download.csv'
        df.to_csv(log_path,index=False)
        print('\t',log_path)


# In[7]:


get_ipython().run_line_magic('time', 'download_data(region,version,lim)')


# In[8]:


get_ipython().system('ls ab_test')


# In[ ]:




