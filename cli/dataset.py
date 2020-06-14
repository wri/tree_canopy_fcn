#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)
from importlib import reload


# In[2]:


import os
import re
import pandas as pd
import image_kit.io as io
import mproc
from glob import glob


# ---

# In[3]:


PRODUCT_NAME='spot'
REGION_NAME='la_built_up_1p5'
RESOLUTION=1.5
MAX_BLACK_PIXEL=512*4


# ---

# In[4]:


DATA_DIR=f'/DATA/imagery/{REGION_NAME}/v1/{RESOLUTION}'
RES_PART=re.sub(r'\.','p',str(RESOLUTION))
DSET_PATH=f'{PROJECT_DIR}/datasets/{REGION_NAME}.{RES_PART}.{PRODUCT_NAME}.STATS.csv'
print('CREATING:',DSET_PATH)


# In[5]:


INPUT_PREFIX='ab_spot'
INPUT_FOLDER='spot'
INPUT_DIR=f'{DATA_DIR}/{INPUT_FOLDER}'
INPUT_PATHS=glob(f'{INPUT_DIR}/*.tif')


# In[6]:


HAS_RGBN=True
RGBN_PREFIX='naip'
RGBN_FOLDER='naip'
RGBN_DIR=f'{DATA_DIR}/{RGBN_FOLDER}'
RGBN_PATHS=glob(f'{RGBN_DIR}/*.tif')


# In[7]:


HAS_LIDAR=True
LIDAR_PREFIX='hag'
LIDAR_FOLDER='lidar/USGS_LPC_CA_LosAngeles_2016_LAS_2018'
LIDAR_DIR=f'{DATA_DIR}/{LIDAR_FOLDER}'
LIDAR_PATHS=glob(f'{LIDAR_DIR}/*.tif')


# ---

# In[8]:


YEAR_TAIL_RGX='_20[0-9]{2}-(train|valid|test).tif$'
TYPE_RGX='-(train|valid|test).'


# ---

# In[9]:


INPUTS_DF=pd.DataFrame(INPUT_PATHS,columns=['input_path'])
print('NB_IMAGES:',INPUTS_DF.shape[0])


# ---

# In[10]:


def get_dset_type(path):
    m=re.search(TYPE_RGX,path)
    if m:
        s,e=m.span()
        return path[s+1:e-1]

    
def stats(im):
    return im.mean(axis=(1,2)), im.std(axis=(1,2))


def get_tile_key(input_path):
    return re.sub(f'^{INPUT_PREFIX}_','',os.path.basename(input_path)).split('_')[0]


def get_path_with_key(paths,key):
    return next((p for p in paths if key in p),False)

    
def get_year(path):
    if path:
        m=re.search(YEAR_TAIL_RGX,path)
        if m:
            s,_=m.span()
            return int(path[s+1:s+5])
    
    
def pair_data(paths,key):
    path=get_path_with_key(paths,key)
    if path:
        im=io.read(path,return_profile=False)
        means,stdevs=stats(im)
    else:
        path=None
        means=None
        stdevs=None
    return path, means, stdevs


def stat_row(row_dict):
    r=row_dict.copy()
    input_path=row_dict['input_path']
    tile_key=get_tile_key(input_path)
    im=io.read(input_path,return_profile=False)
    r['tile_key']=tile_key
    r['means'],r['stdevs']=stats(im)
    r['black_pixel_count']=(im[:3].sum(axis=0)==0).sum()
    r['year']=get_year(input_path)
    r['dset_type']=get_dset_type(input_path)
    if HAS_RGBN:
        path, means, stdevs=pair_data(RGBN_PATHS,tile_key)
        r['rgbn_path']=path
        r['rgbn_means']=means
        r['rgbn_stdevs']=stdevs
        r['rgbn_year']=get_year(path)
    if HAS_LIDAR:
        path, means, stdevs=pair_data(LIDAR_PATHS,tile_key)
        r['hag_path']=path
        r['hag_means']=means
        r['hag_stdevs']=stdevs
    return r


def to_list(arr):
    if (arr is not None) and (arr is not False):
        arr=list(arr)
    return arr


def run_region(lim=None,save=True):
    df=INPUTS_DF.copy().iloc[:lim]
    row_dicts=df.to_dict('records')
    print('NB_IMAGES:',len(row_dicts))
    print(row_dicts[0])
    out=mproc.map_with_threadpool(stat_row,row_dicts,max_processes=64)
    df=pd.DataFrame(out)
    if save:
        _df=df.copy()
        _df['means']=_df.means.apply(to_list)
        _df['stdevs']=_df.stdevs.apply(to_list)
        _df['rgbn_means']=_df.means.apply(to_list)
        _df['rgbn_stdevs']=_df.stdevs.apply(to_list)
        _df['hag_means']=_df.means.apply(to_list)
        _df['hag_stdevs']=_df.stdevs.apply(to_list)
        _df.to_csv(DSET_PATH,index=False)
        return df, DSET_PATH
    else:
        return df


# ---

# In[11]:


df,path=run_region(save=True)
print(path)


# ---

# In[17]:


test=(df.black_pixel_count>MAX_BLACK_PIXEL)
print('NB BLACK > MAX BLACK PIXS:',df[test].shape[0])
_df=df[~test]
print('=>',_df.shape[0])


# In[18]:


print(f'MEANS={_df.means.mean(axis=0).tolist()}')
print(f'STDEVS={_df.stdevs.mean(axis=0).tolist()}')


# In[19]:


pd.read_csv(path,converters={'means': eval}).head()


# In[ ]:




