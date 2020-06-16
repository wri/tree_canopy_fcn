import os,sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)
import pandas as pd
import numpy as np
from utils.dataloader import HeightIndexDataset
import _train_methods as m
# 
# CONSTANTS
# 
DSET='target_data-la_built_up_1p5.1p5.spot.STATS.csv'


#
# HELPERS
#
def badscore(row):
    return row.shadow+row['not-classified']

def to_arr(str_arr):
    return np.array(eval(str_arr))


# 
#  DSET
# 
def to_arr(str_list):
    return np.array(eval(str_list))

CONV={
    'input_means': to_arr,
    'input_stdevs': to_arr,
    'rgbn_means': to_arr,
    'rgbn_stdevs': to_arr,
    'hag_means': to_arr,
    'hag_stdevs': to_arr
}
df=pd.read_csv(f'{PROJECT_DIR}/datasets/{DSET}',converters=CONV)
print('init_size:',df.shape[0])
df['bad_score']=df.apply(badscore,axis=1)
df=df[df.bad_score<0.5]
print('good_size:',df.shape[0])



# # PLIEDES INPUT
m.DSET=df
m.MEANS=df.input_means.mean(axis=0).tolist()
m.STDEVS=df.input_stdevs.mean(axis=0).tolist()
m.INDICES=['ndvi','ndwi']
m.VALUE_MAP={ 8: [9] }
m.SMOOTHING_KERNEL=np.ones((3,3))
m.NB_CATEGORIES=9


def model(**cfig):   
    return m.model(**cfig)


def criterion(**cfig):
    return m.criterion(**cfig)


def optimizer(**cfig):
    return m.optimizer(**cfig)


def loaders(**cfig):
    return m.loaders(**cfig)



