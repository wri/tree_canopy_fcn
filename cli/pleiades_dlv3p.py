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
PLEIADES='PLEIADES-target_data-los_angeles-plieades_naip-lidar_USGS_LPC_CA_LosAngeles_2016_LAS_2018.STATS.csv'
NAIP='NAIP-target_data-los_angeles-plieades_naip-lidar_USGS_LPC_CA_LosAngeles_2016_LAS_2018.STATS.csv'


#
# HELPERS
#
def badscore(row):
    return row.shadow+row.uncertain+row['not-classified']

def to_arr(str_arr):
    return np.array(eval(str_arr))


# 
#  DSET
# 
CONV={
    'naip_means': to_arr,
    'naip_stdevs': to_arr,
    'pleiades_means': to_arr,
    'pleiades_stdevs': to_arr,
}
df=pd.read_csv(f'{PROJECT_DIR}/datasets/{PLEIADES}',converters=CONV)
ndf=pd.read_csv(f'{PROJECT_DIR}/datasets/{NAIP}',converters=CONV)
df['input_path']=df.pleiades_path.copy()
df['naip_bad_score']=ndf.apply(badscore,axis=1)
df=df[df.naip_bad_score<0.255]
df[df.pleiades_year<2018].shape



# # PLIEDES INPUT
m.DSET=df
m.MEANS=df.pleiades_means.mean(axis=0).tolist()
m.STDEVS=df.pleiades_stdevs.mean(axis=0).tolist()
m.INDICES=['ndvi','ndwi']
m.VALUE_MAP={ 8: [9] }
# m.SMOOTHING_KERNEL=np.ones((5,5))
m.NB_CATEGORIES=9


def model(**cfig):   
    return m.model(**cfig)


def criterion(**cfig):
    return m.criterion(**cfig)


def optimizer(**cfig):
    return m.optimizer(**cfig)


def loaders(**cfig):
    return m.loaders(**cfig)



