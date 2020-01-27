import pickle
import re
from config import DL_PREFIX


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




#
# PATH/NAME HELPERS
#
def get_image_id(tile_key,date,prefix=DL_PREFIX):
    date=re.sub('-','',date)
    tile_key=re.sub(':','x',tile_key)
    return f'{prefix}_{tile_key}-{date}'