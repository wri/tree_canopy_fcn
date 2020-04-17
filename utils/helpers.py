from pathlib import Path, PurePath
import pickle
import re
import json
import geojson
import yaml
import numpy as np
from config import DL_PREFIX, VALUE_CATEGORIES, NB_CATS, BATCH_SIZE


#
# UTILS
#
def ensure_dir(path=None,directory=None):
    if path:
        directory=PurePath(path).parent
    Path(directory).mkdir(
            parents=True,
            exist_ok=True)


def save_pickle(obj,path,mkdirs=True):
    """ save object to pickle file
    """ 
    if mkdirs:
        ensure_dir(path)
    with open(path,'wb') as file:
        pickle.dump(obj,file,protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    """ read pickle file
    """    
    with open(path,'rb') as file:
        obj=pickle.load(file)
    return obj


def read_yaml(path,*key_path):
    """ read yaml file
    path<str>: path to yaml file
    *key_path: keys to go to in object
    """    
    with open(path,'rb') as file:
        obj=yaml.safe_load(file)
    for k in key_path:
        obj=obj[k]
    return obj


def read_json(path,*key_path):
    """ read json file
    path<str>: path to json file
    *key_path: keys to go to in object
    """    
    with open(path,'rb') as file:
        obj=json.load(file)
    for k in key_path:
        obj=obj[k]
    return obj


def save_json(obj,path,indent=4,sort_keys=False,mkdirs=True):
    """ save object to json file
    """ 
    if mkdirs:
        ensure_dir(path)
    with open(path,'w') as file:
        json.dump(obj,file,indent=indent,sort_keys=sort_keys)


def read_geojson(path,*key_path):
    """ read geojson file
    path<str>: path to geojson file
    *key_path: keys to go to in object
    """    
    with open(path,'rb') as file:
        obj=geojson.load(file)
    for k in key_path:
        obj=obj[k]
    return obj


def batch_list(lst,batch_size=BATCH_SIZE):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]



#
# PATH/NAME HELPERS
#
def get_image_id(tile_key,date,prefix=DL_PREFIX):
    date=re.sub('-','',date)
    tile_key=re.sub(':','x',tile_key)
    return f'{prefix}_{tile_key}-{date}'




#
# NUMPY
#
def get_histogram(class_band):
    cats,counts=cats_and_counts(class_band)
    hist={ cat: cnt for cat,cnt in zip(cats,counts) }
    return { VALUE_CATEGORIES[cat]: hist.get(cat,0) for cat in range(NB_CATS) }


def cats_and_counts(class_band):
    cats,counts=np.unique(class_band,return_counts=True)
    if isinstance(cats,np.ma.core.MaskedArray):
        cats=cats.data
    cats=[ int(c) for c  in cats ]
    counts=[ int(c) for c  in counts ]
    return cats, counts

