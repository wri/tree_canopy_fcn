import re
from random import shuffle
import utils.helpers as h
from descarteslabs.scenes import DLTile
from config import REGIONS_DIR,TILES_DIR
from config import RESOLUTION, TILESIZE, PAD
import utils.load as load


def save_master_tile_list(
        region_name,
        version=1,
        suffix='master',
        split=True,
        resolution=RESOLUTION,
        tilesize=TILESIZE,
        pad=PAD,
        return_tiles=False,
        return_keys=False):
    dest=load.tile_keys_path(region_name,suffix=suffix,version=version)
    out=_fetch_tiles(
            region_name,
            dest=dest,
            resolution=resolution,
            tilesize=tilesize,
            pad=pad,
            return_tiles=return_tiles,
            return_keys=return_keys)
    if out:
        return dest, out
    else:
        return dest


def split_master_tile_list(
        region_name,
        version=1,
        suffix='master',
        valid_frac=0.2,
        test_frac=0.1,
        save=True):
    master_path=load.tile_keys_path(region_name,suffix=suffix,version=version)
    keys=load.tile_keys(path=master_path)
    out=_split_tile_keys(keys,valid_frac=valid_frac,test_frac=test_frac)
    train_path=_save_split(out[0],master_path,'train',suffix)
    valid_path=_save_split(out[1],master_path,'valid',suffix)
    if test_frac:
        train, valid, test=out
        test_path=_save_split(out[2],master_path,'test',suffix)
        return train_path, valid_path, test_path
    else:
        return train_path, valid_path


def _split_tile_keys(keys,valid_frac=0.2,test_frac=0.1):
    shuffle(keys)
    total=len(keys)
    nb_valid=int(total*valid_frac)
    valid=keys[:nb_valid]
    if test_frac:
        nb_test=int(total*test_frac)
        test=keys[nb_valid:nb_valid+nb_test]
        train=keys[nb_valid+nb_test:]
        return train, valid, test
    else:
        train=keys[nb_valid:]
        return train, valid


def _fetch_tiles(
        region_name,
        dest=None,
        resolution=RESOLUTION,
        tilesize=TILESIZE,
        pad=PAD,
        return_tiles=True,
        return_keys=False):
    study_area=load.study_area(region_name)
    tiles=DLTile.from_shape(
        shape=study_area, 
        resolution=RESOLUTION, 
        tilesize=TILESIZE, 
        pad=PAD)
    if dest or return_keys:
        keys=[t.key for t in tiles]
        if dest:
            h.save_pickle(keys,dest)
    if return_keys:
        return keys
    elif return_tiles:
        return tiles


def _save_split(obj,master_path,split_type,suffix='master'):
    path=re.sub(f'-{suffix}',f'-{split_type}',master_path)
    h.save_pickle(obj,path)
    return path



 