#
# IMPORTS
#
ROOT_PATH='/home/ericp/tree_canopy_fcn/repo'
import sys
sys.path.append(ROOT_PATH)
from pprint import pprint
import pandas as pd
import mproc
import utils.lidar as lidar
import utils.helpers as h
import utils.datasets as datasets
import utils.load as load
import utils.paths as paths
import utils.dlabs as dlabs
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
DEFALUT_VERSION=1

YEAR_START=2015
YEAR_END=2018
DEFALUT_AB_PRODUCT='plieades'
PLEIADES_PRODUCTS=['airbus:oneatlas:phr:v2']
PLEIADES_DIR='pleiades'
PLEIADES_PREFIX='ab_pleiades'
SPOT_PRODUCTS=['airbus:oneatlas:spot:v2']
SPOT_DIR='spot'
SPOT_PREFIX='ab_spot'
INPUT_BANDS=['red', 'green', 'blue', 'nir', 'alpha']
ALPHA_BAND=False
DSET_TYPES=['train','valid','test']
DATA_ROOT='/DATA/imagery'
MAX_PROCESSES=8



#
# DOWNLOAD HELPERS
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


def _ab_meta(product,tile_key,year,dset_type):
    if product=='spot':
        product_prefix=SPOT_PREFIX
        product_dir=SPOT_DIR
        products=SPOT_PRODUCTS
    else:
        product_prefix=PLEIADES_PREFIX
        product_dir=PLEIADES_DIR
        products=PLEIADES_PRODUCTS
    return products, f'{DATA_ROOT}/{product_dir}/{product_prefix}_{tile_key}_{year}-{dset_type}.tif'


def airbus_download_tile_for_year(
        tile_key,
        product,
        year,
        dset_type,
        dry_run=False):
    products,dest=_ab_meta(product,tile_key,year,dset_type)
    h.ensure_dir(dest)
    out=None
    error=False
    error_msg=None
    try:
        out=dlabs.mosaic(        
            tile_key,
            products=products,
            bands=INPUT_BANDS,
            alpha_band=ALPHA_BAND,
            start=f'{year}-01-01',
            end=f'{year+1}-01-01',
            dest=dest,
            dry_run=dry_run)
    except Exception as e:
        error=True
        error_msg=str(e)
    if out or error:
        return {
            'tile_key': tile_key,
            'year': year,
            'path': dest,
            'error': error,
            'error_msg': error_msg}


def airbus_download_tile_in_range(
        tile_key,
        dset_type,
        product,
        year_start=YEAR_START,
        year_end=YEAR_END,
        dry_run=False):
    out=None
    for year in range(year_start,year_end+1):
        out=airbus_download_tile_for_year(
            tile_key,
            product,
            year,
            dset_type,
            dry_run)
        if out: break;
    if not out:
        out={'tile_key': tile_key, 'year': None, 'path': None, 'error': False, 'error_msg': None}
    return out





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



@click.command(
    help='airbus: region_name',
    context_settings=ARG_KWARGS_SETTINGS ) 
@click.argument('region_config',type=str)
@click.option(
    '--product',
    help='plieades or spot',
    default=DEFALUT_AB_PRODUCT,
    type=str)
@click.option(
    '--start',
    help='year start',
    default=YEAR_START,
    type=int)
@click.option(
    '--end',
    help='year end',
    default=YEAR_END,
    type=int)
@click.option(
    '--version',
    help='data version',
    default=DEFALUT_VERSION,
    type=int)
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
def airbus(
        region_config,
        product=DEFALUT_AB_PRODUCT,
        start=YEAR_START,
        end=YEAR_END,
        version=DEFALUT_VERSION,
        lim=None,
        dry_run=False):
    print('PRODUCT:',product)
    aoi_config=load.aoi(region_config)
    print('AOI:')
    pprint(aoi_config)
    for typ in DSET_TYPES:
        print()
        keys=load.tile_keys(region_config,typ,frac=aoi_config['sample_frac'])[:lim]
        print(f'DOWLOADING {typ}({len(keys)}):')
        def _download(key):
            return airbus_download_tile_in_range(
                key,
                typ,
                product,
                start,
                end,
                dry_run)
        out=mproc.map_with_threadpool(_download,keys,max_processes=MAX_PROCESSES)
        df=pd.DataFrame(out)
        log_path=f'{product}-{typ}_download.csv'
        df.to_csv(log_path,index=False)
        print('\t',log_path)





cli.add_command(tilesets)
cli.add_command(lidar)
cli.add_command(airbus)
if __name__ == "__main__":
    cli()

