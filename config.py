import os


#
# DIRECTORIES
#
HOME=os.path.expanduser('~')
PROJECT_DIR=os.path.dirname(os.path.realpath(__file__))
DATA_DIR=f'{PROJECT_DIR}/data'
DATASETS_DIR=f'{PROJECT_DIR}/datasets'
PRODUCTS_DIR=f'{PROJECT_DIR}/products'
REGIONS_DIR=f'{PROJECT_DIR}/regions'
CLI_DIR=f'{PROJECT_DIR}/cli'
TILES_DIR=f'{DATA_DIR}/tiles'


#
# DATA
#
HEIGHT_THRESHOLD=4
NDVI_THRESHOLD=0.1


#
# FILES
#
TILE_MAP_PATH='/datadrive/UTC/Los_Angeles/shp/tile_map.pkl'


#
# TILING
#
RESOLUTION=1.0
PAD=16
SIZE=512
TILESIZE=SIZE-2*PAD


#
# STATS
#

# # # PRODUCT: NAIP18_LA
# MEANS=[
#     101.12673535231546,
#     100.36417761244,
#     94.04471640665643,
#     113.85310697286442]

# STDEVS=[
#     39.0196407883084,
#     35.3287659336378,
#     33.68392945659178,
#     35.37087488392215]

# # PRODUCT: NAIP18_SF
# MEANS=[
#     109.4278706897709,
#     118.19323218294377,
#     120.46183497601471,
#     98.99740405233365]
# STDEVS=[
#     42.19698635459705,
#     38.94906623016931,
#     36.65071656760073,
#     41.29432206181956]


# PRODUCT: AIRBUS_SF
MEANS=[
    82.57092282633093,
    91.45992765706548,
    80.54208946149178,
    104.85985499898872]
STDEVS=[
    41.86997174859683,
    38.974401292962796,
    41.468384512291195,
    35.871153206840034]



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


GREENSPACE_VALUE_CATEGORIES={
    0: 'None',
    1: 'Grass',
    2: 'Shrub',
    3: 'Small Tree',
    4: 'Tree',
    5: 'No Data'
}
GREENSPACE_THRESHOLDS=[0.5,2,4]
NB_GREENSPACE_CATS=len(GREENSPACE_VALUE_CATEGORIES)
GREENSPACE_COLORS={
    'None': '#fcf8e8',
    'Grass': '#ff7b9a',
    'Shrub': '#6f0000',
    'Small Tree': '#add632',
    'Tree': '#306100',
    'No Data': '#ff0000'
}




BUILTUP_COLORS={
    'None': '#fcf8e8',
    'water': '#0000ff',
    # 'test': '#886633',
    'road': '#ff7b9a',
    '1 story': '#6f0000',
    '2-3 storys': '#add632',
    '4+ storys': '#306100',
}
BUILTUP_CATEGORY_THRESHOLDS=[
            { # water  
                'ndwi': 0.35,
            },
            # { # test
            #     'ndwi': [-1e-9,0.05],
            #     'height': [1.6,]
            # },
            { # road
                'ndwi': [0.05,0.35],
                'height': [0,1.6]
            },
            { # 1-story
                'ndwi': [-0.075,0.35],
                'height': [1.6,5]
            },
            { # 2-3 story
                'ndwi': [-0.075,0.35],
                'height': [5,10]
            },
            { # 4+ story
                'ndwi': [-0.075,0.35],
                'height': 10
            }]


#
# OTHER
#
LOCAL_SRC=False
DL_PREFIX='TREECANOPY'
BATCH_SIZE=12
INPUT_BANDS=['red', 'green', 'blue', 'nir', 'alpha']
# # NAIP < 2018
# ALPHA_BAND=False
# PRODUCTS=['usda:naip:rgbn:v1']
# # NAIP 2018
# ALPHA_BAND=True
# PRODUCTS=['usda:nrcs:naip:rgbn:v1']
# # AIRBUS
ALPHA_BAND=True
PRODUCTS=['airbus:oneatlas:phr:v2']



