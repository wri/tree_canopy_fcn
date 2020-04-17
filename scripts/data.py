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
import click
ARG_KWARGS_SETTINGS={
    'ignore_unknown_options': True,
    'allow_extra_args': True
}



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
# HELPERS
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



#
# CLI INTERFACE
#
@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj={}


@click.command(
    help='tilesets: region_name',
    context_settings=ARG_KWARGS_SETTINGS ) 
@click.argument('region_config',type=str)
def tilesets(region_config):
    print(region_config)
    geometry=load.aoi(region_config,'geometry')
    print(geometry)
    # """ save tile keys for region """
    # print('\n'*2)
    # print("master_keys:")
    # master_keys_path,master_keys=datasets.save_master_tile_list(region,return_keys=True)
    # print()
    # print(master_keys_path,len(master_keys),master_keys[0])


    # """ split tile keys into train/valid/test """
    # print('\n'*2)
    # print("split_paths:")
    # print()
    # split_paths=datasets.split_tile_keys(region)
    # pprint(split_paths)


    # """ create samples from train/valid/test sets """
    # print('\n'*2)
    # print("samples:")
    # print()
    # sample_paths=datasets.sample_tile_keys(region,frac=SAMPLE_FRAC)
    # for p in sample_paths:
    #     print(p.split('/')[-1],len(h.read_pickle(p)))


@click.command(
    help='lidar: region_name',
    context_settings=ARG_KWARGS_SETTINGS ) 
@click.argument('region_config',type=str)
@click.option(
    '--lim',
    help='limit for dev',
    default=None,
    type=int)
def lidar(region_config,lim=None):
    aoi=load.aoi(region_config)
    src=f"{aoi['ept_root']}/{aoi['ept_folder']}"
    print(region_config,lim)
    print(aoi)
    print(src)
    # for dset in ['test','valid','train']:
    #     print('\n'*2)
    #     print(f'DOWNLOADING: {dset}')
    #     paths, errors=download_lidar_tiles(
    #         src,
    #         aoi['region'],
    #         dset,
    #         aoi['sample_frac'],
    #         aoi['subdir'],
    #         lim=lim)
    #     print(f'nb_paths: {len(paths)}, nb_errors: {len(errors)}')
    #     if errors:
    #         pprint(errors)



cli.add_command(tilesets)
cli.add_command(lidar)
if __name__ == "__main__":
    cli()

