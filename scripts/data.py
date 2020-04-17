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
import utils.lidar
import click
ARG_KWARGS_SETTINGS={
    'ignore_unknown_options': True,
    'allow_extra_args': True
}



#
# RUN CONFIG
#
USGS_EPT_ROOT="https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
MAX_PROCESSES=32



#
# HELPERS
#
def download_lidar_tiles(
        src,
        region,
        dset,
        frac,
        usgs_folder,
        lim=None,
        max_processes=MAX_PROCESSES,
        dry_run=False):
    keys=load.tile_keys(region,dset,frac=frac)[:lim]
    out=utils.lidar.download_tileset(
        src,
        region,
        keys,
        identifier=dset,
        subdir=usgs_folder,
        max_processes=max_processes,
        dry_run=dry_run)
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
    aoi=load.aoi(region_config)
    region=aoi.get('region',region_config)
    geometry=aoi.get('geometry')
    version=aoi.get('version',1)
    sample_frac=aoi['sample_frac']
    print('TILESETS: ',region,geometry,version)
    """ save tile keys for region """
    print('\n'*2)
    print("master_keys:")
    master_keys_path,master_keys=datasets.save_master_tile_list(
        region,
        geometry=geometry,
        version=version,
        return_keys=True)
    print()
    print(master_keys_path,len(master_keys),master_keys[0])


    """ split tile keys into train/valid/test """
    print('\n'*2)
    print("split_paths:")
    print()
    split_paths=datasets.split_tile_keys(region)
    pprint(split_paths)


    """ create samples from train/valid/test sets """
    print('\n'*2)
    print("samples:")
    print()
    sample_paths=datasets.sample_tile_keys(region,frac=sample_frac)
    for p in sample_paths:
        print(p.split('/')[-1],len(h.read_pickle(p)))


@click.command(
    help='lidar: region_name',
    context_settings=ARG_KWARGS_SETTINGS ) 
@click.argument('region_config',type=str)
@click.option(
    '--lim',
    help='limit for dev',
    default=None,
    type=int)
@click.option(
    '--dry_run',
    help='dry_run',
    default=False,
    type=bool)
def lidar(region_config,lim=None,dry_run=False):
    aoi=load.aoi(region_config)
    src=f"{aoi.get('ept_root',USGS_EPT_ROOT)}/{aoi['ept_folder']}"
    region=aoi.get('region',region_config)
    sample_frac=aoi['sample_frac']
    subdir=aoi['subdir']
    lim=lim or aoi.get('lim')
    max_processes=aoi.get('max_processes',MAX_PROCESSES)
    print('LIDAR: ',region_config,lim)
    pprint(aoi)
    for dset in ['test','valid','train']:
        print('\n'*2)
        print(f'DOWNLOADING: {dset}')
        paths, errors=download_lidar_tiles(
            src,
            region,
            dset,
            sample_frac,
            subdir,
            lim=lim,
            max_processes=max_processes,
            dry_run=dry_run)
        print(f'nb_paths: {len(paths)}, nb_errors: {len(errors)}')
        if errors:
            pprint(errors)



cli.add_command(tilesets)
cli.add_command(lidar)
if __name__ == "__main__":
    cli()

