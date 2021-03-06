import sys
sys.path.append('..')
import os
from pathlib import Path
import re
import pandas as pd
import image_kit.io as io
import mproc
from glob import glob
from pprint import pprint
import click
import utils.load as load

_filedir=os.path.dirname(os.path.realpath(__file__))



#
# CONSTANTS
#
YEAR_TAIL_RGX='_20[0-9]{2}-(train|valid|test).tif$'
TYPE_RGX='-(train|valid|test).'
MAX_BLACK_PIXEL=512*4
DATA_DIR='/DATA/imagery'
VERSION=1
PROJECT_DIR=Path(_filedir).parent


#
# HELPERS
#
def get_dset_type(path):
    m=re.search(TYPE_RGX,path)
    if m:
        s,e=m.span()
        return path[s+1:e-1]

    
def stats(im):
    return list(im.mean(axis=(1,2))), list(im.std(axis=(1,2)))


def get_tile_key(input_path,input_prefix):
    return re.sub(f'^{input_prefix}_','',os.path.basename(input_path)).split('_')[0]


def get_path_with_key(paths,key):
    return next((p for p in paths if key in p),False)

    
def get_year(path):
    if path:
        m=re.search(YEAR_TAIL_RGX,path)
        if m:
            s,_=m.span()
            return int(path[s+1:s+5])
    
    
def im_data(path=None,paths=None,key=None):
    if not path:
        path=get_path_with_key(paths,key)
    if path:
        im=io.read(path,return_profile=False)
        means,stdevs=stats(im)
        black_pixel_count=(im[:3].sum(axis=0)==0).sum()
        shape=list(im.shape[1:])
    else:
        path=False
        means=None
        stdevs=None
        black_pixel_count=None
        shape=None
        im=None
    return path, means, stdevs, black_pixel_count, shape, im


def stat_row(row_dict,input_paths,rgbn_paths,lidar_paths,input_prefix):
    r=row_dict.copy()
    input_path=row_dict['input_path']
    dset_type=get_dset_type(input_path)
    tile_key=get_tile_key(input_path,input_prefix)
    r['dset_type']=dset_type
    r['tile_key']=tile_key
    path,means,stdevs,bpc,shape,_=im_data(path=input_path)
    r['input_path']=path
    r['input_means']=means
    r['input_stdevs']=stdevs
    r['input_black_pixel_count']=bpc
    r['input_shape']=shape
    r['input_year']=get_year(path)
    if rgbn_paths:
        path,means,stdevs,bpc,shape,_=im_data(paths=rgbn_paths,key=tile_key)
        r['rgbn_path']=path
        r['rgbn_means']=means
        r['rgbn_stdevs']=stdevs
        r['rgbn_black_pixel_count']=bpc
        r['rgbn_shape']=shape
        r['rgbn_year']=get_year(path)
    if lidar_paths:
        path,means,stdevs,bpc,shape,im=im_data(paths=lidar_paths,key=tile_key)
        r['hag_path']=path
        r['hag_means']=means
        r['hag_stdevs']=stdevs
        r['hag_shape']=shape
        r['hag_ground']=bpc
        if im is None:
            r['hag_negative']=None
        else:
            r['hag_negative']=(im<0).sum()
    return r


def to_list(arr):
    if (arr is not None) and (arr is not False):
        arr=list(arr)
    return arr


def run_region(inputs_dir,rgbn_dir,lidar_dir,dset_path,input_prefix):
    input_paths=glob(f'{inputs_dir}/*.tif')
    if rgbn_dir:
        rgbn_paths=glob(f'{rgbn_dir}/*.tif')
    else:
        rgbn_paths=False
    if lidar_dir:
        lidar_paths=glob(f'{lidar_dir}/*.tif')
    else:
        lidar_paths=False
    print(inputs_dir,len(input_paths))
    print(rgbn_dir,len(rgbn_paths))
    print(lidar_dir,len(lidar_paths))
    df=pd.DataFrame(input_paths,columns=['input_path'])
    row_dicts=df.to_dict('records')
    print('nb_inputs:',len(row_dicts))
    def _stat_row(row_dict):
        return stat_row(row_dict,input_paths,rgbn_paths,lidar_paths,input_prefix)
    out=mproc.map_with_threadpool(_stat_row,row_dicts,max_processes=64)
    df=pd.DataFrame(out)    
    print('complete:',df.shape)
    print('clean-up:')
    df=df[df.input_black_pixel_count<MAX_BLACK_PIXEL]
    print('.',df.shape[0])
    print('.. - hag ok?',df.hag_negative.mean())
    # df=df[df.hag_negative<(MAX_BLACK_PIXEL*3)]  
    # print('..',df.shape[0])
    print('hag',df.hag_shape.astype(str).unique())
    print('rgbn',df.rgbn_shape.astype(str).unique())
    print('input',df.input_shape.astype(str).unique())
    df=df[df.hag_shape.apply(lambda s: s==[512,512])] 
    print('...',df.shape[0])
    df=df[df.rgbn_shape.apply(lambda s: s==[512,512])]    
    print('....',df.shape[0])
    df=df[df.input_shape.apply(lambda s: s==[512,512])]
    print('----',df.shape[0])
    print('sample:\n')
    print(df.sample().iloc[0])
    df.to_csv(dset_path,index=False)
    print()
    return dset_path


#
# RUN
#

@click.command( help='generate dataset' )
@click.argument('region_name',type=str)
def run(region_name):
    print('REGION:',region_name)
    aoi=load.aoi(region_name)
    root_dir=f'{DATA_DIR}/{region_name}'
    root_dir=f'{root_dir}/v{aoi.get("version",VERSION)}/{aoi["resolution"]}'
    inputs_dir=f'{root_dir}/{aoi["input_folder"]}'
    rgbn_folder=aoi.get('rgbn_folder')
    if rgbn_folder:
        rgbn_dir=f'{root_dir}/{rgbn_folder}'
    else:
        rgbn_dir=None
    lidar_folder=aoi.get('lidar_folder')
    if lidar_folder:
        lidar_dir=f'{root_dir}/{lidar_folder}'
        subdir=aoi.get('subdir')
        if subdir:
            lidar_dir=f'{lidar_dir}/{subdir}'
    else:
        lidar_dir=None
    input_product=aoi["input_product"]
    input_prefix=aoi.get('input_prefix',f'ab_{input_product}')
    res_part=re.sub(r'\.','p',str(aoi['resolution']))
    dset_path=f'{PROJECT_DIR}/datasets/{region_name}.{res_part}.{input_product}.STATS.csv'
    path=run_region(inputs_dir,rgbn_dir,lidar_dir,dset_path,input_prefix)
    print('DSET:',path)



if __name__ == "__main__":
    run()


