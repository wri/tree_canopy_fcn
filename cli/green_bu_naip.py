import os,sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)
from utils.dataloader import HeightIndexDataset

import _train_methods as m


# NAIP STATS: ALL (<2017)
m.MEANS=[106.47083152919251, 104.25520495313522, 98.61836143687523, 119.95594400425841]
m.STDEVS=[38.23711386806666, 34.410688920150264, 31.468324931640534, 31.831786730471276]
m.DSETS_PATH=f'{PROJECT_DIR}/datasets/los_angeles-plieades_naip-lidar_USGS_LPC_CA_LosAngeles_2016_LAS_2018.STATS.csv'
m.YEAR_MAX=2021
m.IBNDS=None
m.CAT_BOUNDS=HeightIndexDataset.NAIP_ALL
m.INDICES=['ndvi','ndwi']
m.TARGET_RGBN=False
m.TARGET_RGBN_AS_INPUT=True


def model(**cfig):   
    return m.model(**cfig)


def criterion(**cfig):
    return m.criterion(**cfig)


def optimizer(**cfig):
    return m.optimizer(**cfig)


def loaders(**cfig):
    return m.loaders(**cfig)



