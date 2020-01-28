import sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)
import utils.helpers as h
import utils.load as load
import utils.predict as predict


#
# CONFIG
#
REGION_NAME='la_plus'
START=None
END=20
YEAR=2018
DATE=f'{YEAR}-07-01'
MODEL_NAME='shallow-classifier_after-aspp_false-os_4-ss_2'
LOCAL_SRC=False
DEV=False

if DEV:
    PRODUCT_ID='wri:treecanopy'
    LIM=8
    BATCH_SIZE=3
    NOISE_REDUCER=None
else:
    PRODUCT_ID='wri:treecanopy'
    LIM=None
    BATCH_SIZE=12
    NOISE_REDUCER=10

#
# TILES
#
TILE_KEYS=list(load.TILE_MAP.keys())[:LIM]
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
        if NOISE_REDUCER and (not (i%NOISE_REDUCER)): 
            print('\t-',i)
        predict.descartes_run(
            product_id=PRODUCT_ID,
            model=model,
            batch_keys=batch_keys,
            date=DATE,
            bands=BANDS_META,
            year=YEAR,
            local_src=LOCAL_SRC)


run(BATCHED_KEYS,START,END)



