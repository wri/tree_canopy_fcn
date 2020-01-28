import os


#
# DIRECTORIES
#
HOME=os.path.expanduser('~')
PROJECT_DIR=os.path.dirname(os.path.realpath(__file__))
DATA_DIR=f'{PROJECT_DIR}/data'
PRODUCTS_DIR=f'{PROJECT_DIR}/products'
CLI_DIR=f'{PRODUCTS_DIR}/cli'
TILES_DIR=f'{DATA_DIR}/tiles'


#
# FILES
#
TILE_MAP_PATH='/datadrive/UTC/Los_Angeles/shp/tile_map.pkl'



#
# STATS
#
MEANS=[
    101.12673535231546,
    100.36417761244,
    94.04471640665643,
    113.85310697286442]

STDEVS=[
    39.0196407883084,
    35.3287659336378,
    33.68392945659178,
    35.37087488392215]


#
# MODEL
#
DEFAULT_MODEL_TYPE='dlv3p'
DEFAULT_NB_INPUT_CH=5
MODEL_CONFIG_FILE='treecover'


#
# DATA
#
VALUE_CATEGORIES={
    0: 'Not Tree',
    1: 'Tree',
    2: 'No Data',
}
NB_CATS=len(VALUE_CATEGORIES)


#
# OTHER
#
SAVE_LOCAL=False
DL_PREFIX='TREECANOPY'
BATCH_SIZE=12
INPUT_BANDS=['red', 'green', 'blue', 'nir', 'alpha']
PRODUCTS=['usda:nrcs:naip:rgbn:v1']



