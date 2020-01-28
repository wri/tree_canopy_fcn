import pickle
import re
import geojson
import numpy as np
from config import DL_PREFIX, VALUE_CATEGORIES, NB_CATS, BATCH_SIZE


#
# UTILS
#
def save_pickle(obj,path):
    """ save object to pickle file
    """    
    with open(path,'wb') as file:
        pickle.dump(obj,file,protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    """ read pickle file
    """    
    with open(path,'rb') as file:
        obj=pickle.load(file)
    return obj


def read_geojson(path):
    with open(path) as file:
        data=geojson.load(file)
    return data


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

