import os,sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)
from utils.dataloader import HeightIndexDataset

import _train_methods as m


# # PLIEDES INPUT
MEANS=[94.79936157686979, 92.8912348691044, 80.50194782393349, 108.14889758142212]
STDEVS=[36.37876660224377, 33.22686387734999, 33.30808192430284, 30.075380846943716]
m.DSETS_PATH=f'{PROJECT_DIR}/datasets/los_angeles-plieades_naip-lidar_USGS_LPC_CA_LosAngeles_2016_LAS_2018.STATS.csv'
m.YEAR_MAX=2021
m.IBNDS=None
m.CAT_BOUNDS=HeightIndexDataset.NAIP_ALL
m.INDICES=['ndvi','ndwi']
m.TARGET_RGBN=True
m.TARGET_RGBN_AS_INPUT=False


def model(**cfig):   
    return m.model(**cfig)


def criterion(**cfig):
    return m.criterion(**cfig)


def optimizer(**cfig):
    return m.optimizer(**cfig)


def loaders(**cfig):
    return m.loaders(**cfig)



