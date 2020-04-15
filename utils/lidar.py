import json
import pdal
from descarteslabs.scenes import DLTile
import pyproj
from shapely.ops import transform
from config import RESOLUTION
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



#
# PUBLIC
#
def download_tile(src,dest,tile,**kwargs):
    if isinstance(tile,str):
        tile=DLTile.from_key(tile)
    return download(
        src,
        dest,
        crs=tile.crs,
        crs_bounds=tile.bounds,
        ept_bounds=transform(EPT_PROJECTOR.transform,tile.geometry).bounds,
        **kwargs)


def download(
        src=None,
        dest=None,
        crs=None,
        crs_bounds=None,
        ept_bounds=None,
        pline=None,
        **kwargs):
    if not pline:
        pline=pipeline(src,dest,crs,crs_bounds,ept_bounds,**kwargs)
    pline=pdal.Pipeline(json.dumps(pline))
    pline.validate() 
    pline.execute()


def pipeline(
        src,
        dest,
        crs,
        crs_bounds,
        ept_bounds,
        resolution=RESOLUTION,
        height_bounds=HEIGHT_BOUNDS,
        groundify=True,
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
        value=default
    if value:
        pline.append(value)
    return pline
