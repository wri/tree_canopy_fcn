import os.path
import multiprocessing as mp
from pathlib import Path
from pprint import pprint
import json
import pdal
from descarteslabs.scenes import DLTile
import pyproj
from shapely.ops import transform
import mproc
from config import RESOLUTION
import utils.paths as paths
#
# CONSTANTS/DEFAULTS
#
EPSG4326='epsg:4326'
EPT_SRS='EPSG:3857'
EPT_PROJECTOR=pyproj.Transformer.from_proj(
    pyproj.Proj(init=EPSG4326),
    pyproj.Proj(init=EPT_SRS))

RESOLUTION=1.0
HEIGHT_BOUNDS=[-5,1000]
GROUNDIFY={ "type": "filters.smrf" }
HAG={ "type": "filters.hag" } 
OUTLIERS={   
    "type": "filters.outlier",
    "method": "statistical",
    "multiplier": 3,
    "mean_k": 8
}
GDAL_OUTPUT_TYPE="mean"
GDAL_WINDOW_SIZE=3
MAX_PROCESSES=32


#
# PUBLIC
#
def download_tileset(
        src,
        region,
        keys,
        prefix='hag',
        version=1,
        subdir=None,
        identifier=None,
        max_processes=MAX_PROCESSES,
        dry_run=False,
        overwrite=False):
    tile_dir=paths.lidar_tile(
            region,
            tile_key=False,
            prefix=prefix,
            version=version,
            subdir=subdir,
            identifier=identifier)
    Path(tile_dir).mkdir(parents=True,exist_ok=True)
    def _download(key):
        dest=paths.lidar_tile(
            region,
            tile_key=key,
            prefix=prefix,
            version=version,
            subdir=subdir,
            identifier=identifier)
        if (not overwrite) and os.path.isfile(dest):
            return dest
        else:
            try:
                return download_tile(src,dest,key,dry_run)
            except Exception as e:
                return f'ERROR ({dest}): {e}'
    if max_processes>1:
        return mproc.map_with_threadpool(_download,keys,max_processes=max_processes)
    else:
        print('WARNING: MULTI-PROCESSING OFF')
        return [_download(k) for k in keys]

""" map-with-pool-experiment
MAX_PROCESSES=mp.cpu_count()-1
print(MAX_PROCESSES)
def download_tileset(
        src,
        region,
        keys,
        prefix='hag',
        version=1,
        subdir=None,
        identifier=None,
        max_processes=MAX_PROCESSES,
        overwrite=False):
    tile_dir=paths.lidar_tile(
            region,
            tile_key=False,
            prefix=prefix,
            version=version,
            subdir=subdir,
            identifier=identifier)
    Path(tile_dir).mkdir(parents=True,exist_ok=True)
    args_list=_args_list(keys,src,region,prefix,version,subdir,identifier,overwrite)
    if max_processes>1:
        return mproc.map_with_pool(_download,args_list,max_processes=max_processes)
    else:
        print('WARNING: MULTI-PROCESSING OFF')
        return [_download(k) for k in keys]


def _args_list(values,*args):
    args=list(args)
    return [ [v]+args for v in values ]


def _download(args):
    key,src,region,prefix,version,subdir,identifier,overwrite=args
    dest=paths.lidar_tile(
        region,
        tile_key=key,
        prefix=prefix,
        version=version,
        subdir=subdir,
        identifier=identifier)
    if (not overwrite) and os.path.isfile(dest):
        return dest
    else:
        try:
            return download_tile(src,dest,key)
        except Exception as e:
            return f'ERROR ({dest}): {e}'
"""


def download_tile(src,dest,tile,dry_run=False,**kwargs):
    if isinstance(tile,str):
        tile=DLTile.from_key(tile)
    return download(
        src,
        dest,
        crs=tile.crs,
        crs_bounds=tile.bounds,
        ept_bounds=transform(EPT_PROJECTOR.transform,tile.geometry).bounds,
        dry_run=dry_run,
        **kwargs)


def download(
        src=None,
        dest=None,
        crs=None,
        crs_bounds=None,
        ept_bounds=None,
        pline=None,
        dry_run=False,
        **kwargs):
    if not pline:
        pline=pipeline(src,dest,crs,crs_bounds,ept_bounds,**kwargs)
    _pline=pline
    pline=pdal.Pipeline(json.dumps(pline))
    pline.validate() 
    if dry_run:
        print('\n'*2)
        print(_pline)
        print('\n'*2)
    else:
        pline.execute()
    return dest


def pipeline(
        src,
        dest,
        crs,
        crs_bounds,
        ept_bounds,
        resolution=RESOLUTION,
        height_bounds=HEIGHT_BOUNDS,
        groundify=False,
        hag=True,
        outliers=True):
    pline=[{   
        "type": "readers.ept",
        "filename": src,
        "spatialreference": EPT_SRS,
        "bounds": _bounds_str(ept_bounds) }]
    pline=_update_pipeline(pline,groundify,GROUNDIFY)
    pline=_update_pipeline(pline,hag,HAG)
    pline=_update_pipeline(pline,outliers,OUTLIERS)
    limits="Classification![7:7]"
    if height_bounds:
        limits=f"{limits},HeightAboveGround[{height_bounds[0]}:{height_bounds[1]}]"
    pline=_update_pipeline(pline,{ "type": "filters.range", "limits": limits})
    pline=_update_pipeline(pline,{   
            "type": "filters.reprojection",
            "in_srs": EPT_SRS,
            "out_srs": crs })
    pline=_update_pipeline(pline,{   
            "type": "filters.crop",
            "bounds": _bounds_str(crs_bounds)
        })
    pline=_update_pipeline(pline,{   
            "filename": dest,
            "dimension":'HeightAboveGround',
            "output_type": GDAL_OUTPUT_TYPE,
            "gdaldriver": "GTiff",
            "resolution": str(resolution),
            "window_size": str(GDAL_WINDOW_SIZE),
            "type": "writers.gdal"
        })
    return { 'pipeline': pline }



#
# INTERNAL
#
def _bounds_str(bounds):
    if isinstance(bounds,(list,tuple)):
        bounds=str(([bounds[0], bounds[2]],[bounds[1], bounds[3]]))
    return bounds


def _update_pipeline(pline,value,default=None):
    if value and (value is True):
        value=default.copy()
    if value:
        pline.append(value)
    return pline
