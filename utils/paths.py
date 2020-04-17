from config import PRODUCTS_DIR, IMAGERY_ROOT_DIR
from config import TILES_DIR, GEOMETRY_DIR, AOI_DIR
import descarteslabs as dl



def aoi(region_config):
    return f'{AOI_DIR}/{region_config}.yaml'


def study_area(geometry_name):
    return f'{GEOMETRY_DIR}/{geometry_name}.geojson'


def tile_keys(region_name,suffix=None,version=1,frac=None):
    path=f'{TILES_DIR}/{region_name}'
    if suffix:
        path=f'{path}-{suffix}'
    if version:
        path=f'{path}.v{version}'
    if frac:
        path=f'{path}-s{int(frac*100)}'
    return f'{path}.p'


def lidar_tile(
        region_name,
        tile_key=None,
        prefix='hag',
        version=1,
        subdir=None,
        identifier=None):
    if isinstance(tile_key,dl.scenes.geocontext.DLTile):
        tile_key=tile_key.key
    path=f'{IMAGERY_ROOT_DIR}/{region_name}'
    if version:
        path=f'{path}/v{version}'
    path=f'{path}/lidar'
    if subdir:
        path=f'{path}/{subdir}'
    if tile_key:
        path=f'{path}/{prefix}_{tile_key}'
        if identifier:
            path=f'{path}-{identifier}'
        return f'{path}.tif'
    else:
        return path