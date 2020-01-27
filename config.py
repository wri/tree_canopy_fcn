import os


#
# DIRECTORIES
#
HOME=os.path.expanduser('~')
PROJECT_DIR=os.path.dirname(os.path.realpath(__file__))
PRODUCTS_DIR='{}/products'.format(PROJECT_DIR)
CLI_DIR='{}/cli'.format(PROJECT_DIR)


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
# OTHER
#
SAVE_LOCAL=False
DL_PREFIX='TREECANOPY'