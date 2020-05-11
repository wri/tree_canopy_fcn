import sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)
import utils.helpers as h
import utils.load as load
import utils.predict as predict


#
# CONFIG
#
# REGION_NAME='la_plus'
REGION_NAME='san_francisco'
START=None
END=None
YEAR=2018
# # NAIP
# INPUT_START_DATE=None
# INPUT_END_DATE=None
# AIRBUS
INPUT_START_DATE=f'{YEAR}-01-01'
INPUT_END_DATE=f'{YEAR}-09-01'
DATE=f'{YEAR}-07-01'
MODEL_NAME='shallow-classifier_after-aspp_false-os_4-ss_2'
LOCAL_SRC=False
DEV=True

if DEV:
    DATE=f'2003-07-01'
    PRODUCT_ID='wri:dev_1'
    LIM=100
    BATCH_SIZE=12
    NOISE_REDUCER=None
else:
    PRODUCT_ID='wri:treecanopy'
    LIM=None
    BATCH_SIZE=12
    NOISE_REDUCER=10

#
# TILES
#

""" RUN ARCHIVE
# - INIT RUN
TILE_KEYS=list(load.TILE_MAP.keys())[:LIM]
# - 2nd RUN
INIT_TILE_KEYS=list(load.TILE_MAP.keys())
REGION_TILE_KEYS=load.tile_keys(REGION_NAME)
TILE_KEYS=[k for k in REGION_TILE_KEYS if k not in INIT_TILE_KEYS]
"""
# FULL RUN
TILE_KEYS=load.tile_keys(REGION_NAME)[:LIM]
BATCHED_KEYS=h.batch_list(TILE_KEYS,batch_size=BATCH_SIZE)


#
# MODEL
#
WEIGHTS_LIST=load.available_weights(MODEL_NAME)
BANDS_META=load.meta('dev','bands')
model=load.model(MODEL_NAME,init_weights=WEIGHTS_LIST[-1])


#
# RUN
#
def run(batches,start=None,end=None):
    nb_batches=len(batches[start:end])
    print('BATCH SIZE:',BATCH_SIZE)
    print('NB TILES:',len(TILE_KEYS))
    print('NB BATCHES:',len(BATCHED_KEYS))
    print(f'RUN[{nb_batches}]:')
    for i,batch_keys in enumerate(batches[start:end]):
        if (not NOISE_REDUCER) or (not (i%NOISE_REDUCER)): 
            print('\t-',i)
        predict.descartes_run(
            product_id=PRODUCT_ID,
            model=model,
            batch_keys=batch_keys,
            date=DATE,
            bands=BANDS_META,
            year=YEAR,
            start=INPUT_START_DATE,
            end=INPUT_END_DATE,
            local_src=LOCAL_SRC)


run(BATCHED_KEYS,START,END)





from time import sleep
import os
print('\n'*10)
print('-'*50)
print('-'*50)
print('turning off...')
sleep(60*5)
print('goodbye')
print('-'*50)
print('-'*50)
print('\n'*10)
os.system('sudo poweroff')



