import sys
PROJECT_DIR='/home/ericp/tree_canopy_fcn/repo'
sys.path.append(PROJECT_DIR)
import utils.load as load
import utils.predict as predict


#
# CONFIG
#
BATCH_SIZE=12
PRODUCT_ID='wri:tree_canopy'
DATE='2016-07-01'
MODEL_NAME='shallow-classifier_after-aspp_false-os_4-ss_2'
START=90
END=None
NOISE_REDUCER=10

#
# TILES
#
TILE_KEYS=list(load.TILE_MAP.keys())
BATCHED_KEYS=[TILE_KEYS[i:i+BATCH_SIZE] for i in range(0, len(TILE_KEYS), BATCH_SIZE)]


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
        if not (i%NOISE_REDUCER): 
            print('\t-',i)
        predict.descartes_run(
            product_id=PRODUCT_ID,
            model=model,
            batch_keys=batch_keys,
            date=DATE,
            bands=BANDS_META)


run(BATCHED_KEYS,START,END)



