#
# IMPORTS
#
ROOT_PATH='/home/ericp/tree_canopy_fcn/repo'
import sys
sys.path.append(ROOT_PATH)
from pprint import pprint
import mproc
import utils.lidar as lidar
import utils.helpers as h
import utils.datasets as datasets
import utils.load as load
import utils.paths as paths
import utils.lidar as lidar



#
# RUN CONFIG
#
LIM=4
FETCH_TILES=False
REGION="LA-dev"
USGS_FOLDER="USGS_LPC_CA_LosAngeles_2016_LAS_2018"
SAMPLE_FRAC=0.2
USGS_ROOT="https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
EPT_URL=f"{USGS_ROOT}/{USGS_FOLDER}"
MAX_PROCESSES=1




#
# DATA PREP
#
if FETCH_TILES:
    """ save tile keys for region """
    print('\n'*2)
    print("master_keys:")
    master_keys_path,master_keys=datasets.save_master_tile_list(REGION,return_keys=True)
    print()
    print(master_keys_path,len(master_keys),master_keys[0])


    """ split tile keys into train/valid/test """
    print('\n'*2)
    print("split_paths:")
    print()
    split_paths=datasets.split_tile_keys(REGION)
    pprint(split_paths)


    """ create samples from train/valid/test sets """
    print('\n'*2)
    print("samples:")
    print()
    sample_paths=datasets.sample_tile_keys(REGION,frac=SAMPLE_FRAC)
    for p in sample_paths:
        print(p.split('/')[-1],len(h.read_pickle(p)))



#
# DOWNLOAD LIDAR
#
def download_lidar_tiles(src,region,dset,frac,usgs_folder,lim=None):
    keys=load.tile_keys(region,dset,frac=frac)[:lim]
    out=lidar.download_tileset(
        src,
        region,
        keys,
        identifier=dset,
        subdir=usgs_folder,
        max_processes=MAX_PROCESSES)
    paths=[p for p in out if 'ERROR' not in p]
    errors=[p for p in out if 'ERROR' in p]
    return paths, errors


""" run: test, valid, train """
for dset in ['test','valid','train']:
    print('\n'*2)
    print(f'DOWNLOADING: {dset}')
    paths, errors=download_lidar_tiles(EPT_URL,REGION,dset,SAMPLE_FRAC,USGS_FOLDER,lim=LIM)
    print(f'nb_paths: {len(paths)}, nb_errors: {len(errors)}')
    if errors:
        pprint(errors)





