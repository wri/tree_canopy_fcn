import os


#
# DIRECTORIES
#
HOME=os.path.expanduser('~')
PROJECT_DIR=os.path.dirname(os.path.realpath(__file__))
IMAGERY_ROOT_DIR='/DATA/imagery'
DATA_DIR=f'{PROJECT_DIR}/data'
DATASETS_DIR=f'{PROJECT_DIR}/datasets'
PRODUCTS_DIR=f'{PROJECT_DIR}/products'
GEOMETRY_DIR=f'{PROJECT_DIR}/geometries'
AOI_DIR=f'{PROJECT_DIR}/aois'
CLI_DIR=f'{PROJECT_DIR}/cli'
TILES_DIR=f'{DATA_DIR}/tiles'


#
# DATA
#



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
# NAIP BOUNDS
#
NAIP_WATER_BOUNDS=[   
    {
        'category': 'water',
        # 'value': 1,
        'hex': '#56CEFB',
        'ndwi': 0.32,
    }]


NAIP_OPENSPACE_BOUNDS=[   
    {
        'category': 'openspace',
        # 'value': 2,
        'hex':'#F9FC06',
        'ndvi': 0.025,
        'height': {'max': 0.5}
    }
]


NAIP_GROUND_BOUNDS=[   
    {
        'category': 'bare_ground',
        # 'value': 2,
        'hex':'#900C3F',
        'ndvi': [0.025,0.15],
        'height': {'max': 0.5}
    }
]


NAIP_GRASS_BOUNDS=[   
    {
        'category': 'grass',
        # 'value': 2,
        'hex':'#F9FC06',
        'ndvi': 0.2,
        'height': {'max': 0.5}
    }
]

NAIP_GREEN_BOUNDS=[   
    {
        'category': 'shrub',
        # 'value': 3,
        'hex':'#fd7d37',
        'ndvi': 0.2,
        'height': [0.5,2]
    },
    {
        'category': 'tree',
        # 'value': 4,
        'hex': '#91d372', ##00ff00',
        'ndvi': 0.25,
        'height': [2,4]
    },
    {
        'category': 'big-tree',
        # 'value': 5,
        'hex': '#496648',
        'ndvi': 0.25,
        'height': 4
    }
]


NAIP_BU_BOUNDS=[   
    {
        'category': 'road',
        # 'value': 6,
        'hex': '#ffffff',
        'ndwi': [-0.01, 0.15],
        'height': {'max': 1.6}
    },
    {
        'category': '1-story',
        # 'value': 7,
        'hex': '#FFC0CB',
        'ndwi': [-0.2, 0.15],
        'height': [1.6,5]
    },
    {
        'category': '2to3-story',
        # 'value': 8,
        'hex': '#c1cefa',
        'ndwi': [-0.2, 0.25],
        'height': [5,10]
    },
    {
        'category': '4+-story',
        # 'value': 9,
        'hex': '#ff00ff',
        'ndwi': [-0.2, 0.25],
        'height': 10
    }
]


#
# PLEIADES BOUNDS
#
PLEIADES_WATER_BOUNDS=[   
    {
        'category': 'water',
        # 'value': 1,
        'hex': '#56CEFB',
        'ndwi': 0.4,
    }]


PLEIADES_OPENSPACE_BOUNDS=[   
    {
        'category': 'openspace',
        # 'value': 2,
        'hex':'#F9FC06',
        'ndvi': 0.05,
        'height': {'max': 0.5}
    }
]

PLEIADES_GROUND_BOUNDS=[   
    {
        'category': 'bare_ground',
        # 'value': 2,
        'hex':'#900C3F',
        'ndvi': [-0.1,-0.0025],
        'height': {'max': 0.5}
    }
]

PLEIADES_GRASS_BOUNDS=[  
    {
        'category': 'grass',
        # 'value': 2,
        'hex':'#F9FC06',
        'ndvi': 0.15,
        'height': {'max': 0.5}
    }
]

PLEIADES_GREEN_BOUNDS=[   
    {
        'category': 'shrub',
        # 'value': 3,
        'hex':'#fd7d37',
        'ndvi': 0.15,
        'height': [0.5,2]
    },
    {
        'category': 'tree',
        # 'value': 4,
        'hex': '#91d372', ##00ff00',
        'ndvi': 0.15,
        'height': [2,4]
    },
    {
        'category': 'big-tree',
        # 'value': 5,
        'hex': '#496648',
        'ndvi': 0.15,
        'height': 4
    }
]


PLEIADES_BU_BOUNDS=[   
    {
        'category': 'road',
        # 'value': 6,
        'hex': '#ffffff',
        'ndwi': { 'min': -0.15, 'max': 0.35 },
        'height': {'max': 1.6}
    },
    {
        'category': '1-story',
        # 'value': 7,
        'hex': '#FFC0CB',
        'ndwi': { 'min': -0.35 },
        'height': [1.6,5]
    },
    {
        'category': '2to3-story',
        # 'value': 8,
        'hex': '#c1cefa',
        'ndwi': { 'min': -0.35 },
        'height': [5,10]
    },
    {
        'category': '4+-story',
        # 'value': 9,
        'hex': '#ff00ff',
        'ndwi': { 'min': -0.35 },
        'height': 10
    }
]

CATEGORY_BOUNDS={
    'naip_water': NAIP_WATER_BOUNDS,
    'naip_openspace': NAIP_OPENSPACE_BOUNDS,
    'naip_ground': NAIP_GROUND_BOUNDS,
    'naip_grass': NAIP_GRASS_BOUNDS,
    'naip_green': NAIP_GREEN_BOUNDS,
    'naip_bu': NAIP_BU_BOUNDS,
    'pleiades_water': PLEIADES_WATER_BOUNDS,
    'pleiades_openspace': PLEIADES_OPENSPACE_BOUNDS,
    'pleiades_ground': PLEIADES_GROUND_BOUNDS,
    'pleiades_grass': PLEIADES_GRASS_BOUNDS,
    'pleiades_green': PLEIADES_GREEN_BOUNDS,
    'pleiades_bu': PLEIADES_BU_BOUNDS
}


#
# OLD DATA/BOUNDS
#
""" NAIP (v1)

# TREE/NO-TREE

    * NDVI 0.1, HAG=4

# GREENSPACE

    * NDVI 0.1, HAG=[0.5,2,4]

# BU

    { # water  
        'ndwi': 0.35,
    },
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
    }

# AB Experiments
ABP_CATEGORY_BOUNDS=[   
    # {
    #     'category': 'water',
    #     # 'value': 1,
    #     'hex': '#56CEFB',
    #     'ndwi': 0.35,
    # },
    {
        'category': 'grass',
        # 'value': 2,
        'hex':'#ffff00',
        'ndvi': 0.2,
        'height': {'max': 0.5}
    },
    {
        'category': 'shrub',
        # 'value': 3,
        'hex':'#cc9900',
        'ndvi': 0.2,
        'height': [0.5,2]
    },
    {
        'category': 'tree',
        # 'value': 4,
        'hex':'#00ff00',
        'ndvi': 0.25,
        'height': [2,4]
    },
    {
        'category': 'big-tree',
        # 'value': 5,
        'hex': '#006400',
        'ndvi': 0.25,
        'height': 4
    }
    # ,
    # {
    #     'category': 'road',
    #     # 'value': 6,
    #     'hex': '#ffffff',
    #     'ndvi': {'max': 0},
    #     'height': {'max': 1.6}
    # },
    # {
    #     'category': '1-story',
    #     # 'value': 7,
    #     'hex': '#6600ff',
    #     'ndvi': {'max': 0},
    #     'height': [1.6,5]
    # },
    # {
    #     'category': '2to3-story',
    #     # 'value': 8,
    #     'hex': '#ff0000',
    #     'ndvi': {'max': 0},
    #     'height': [5,10]
    # },
    # {
    #     'category': '4+-story',
    #     # 'value': 9,
    #     'hex': '#ff00ff',
    #     'ndvi': {'max': 0},
    #     'height': 10
    # }
]


HEIGHT_THRESHOLD=4
NDVI_THRESHOLD=0.1
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
"""


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



