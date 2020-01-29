from __future__ import print_function
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
from pprint import pprint
import descarteslabs as dl
from descarteslabs.scenes import search, Scene, SceneCollection, DLTile
from descarteslabs.client.services.catalog import Catalog
from descarteslabs.catalog import Image, Product, OverviewResampler
from descarteslabs.catalog import Product
from descarteslabs.catalog import SpectralBand, MicrowaveBand, MaskBand
from descarteslabs.catalog import ClassBand, GenericBand
from config import INPUT_BANDS, PRODUCTS, ALPHA_BAND


#
# CONSTANTS
#
SPECTRAL='spectral'
MICROWAVE='microwave'
MASK='mask'
CLASS='class'


DATE='date'
DATE_PROPERTIES=[
    "properties.date.year",
    "properties.date.month",
    "properties.date.day" ]


OVERVIEWS=[2,4,6,10,12,16,20,24,28,32,36,40,50,60,100,200]
OVERVIEW_RESAMPLER=OverviewResampler.MODE

#
# HELPERS 
#
def raster_info(aoi,bands):
    if _is_str(aoi):
        aoi=DLTile.from_key(aoi)
    meta=_coordinate_info(aoi).copy()
    meta['bands']=bands
    return meta



#
# IMAGERY
#
def mosaic(
        aoi,
        year,
        products=PRODUCTS,
        bands=INPUT_BANDS,
        alpha_band=ALPHA_BAND,
        start=None,
        end=None,
        return_geocontext=False):
    if _is_str(aoi):
        aoi=DLTile.from_key(aoi)
    start=start or f'{year}-05-01'
    end=end or f'{year}-09-01'
    if not alpha_band:
        bands=bands[:4]
    sc,_=search(aoi,products=products,start_datetime=start,end_datetime=end)
    if sc:
        im=sc.mosaic(bands,aoi,mask_nodata=False,raster_info=False)
    else:
        im=False
    if return_geocontext:
        return im, aoi
    else:
        return im




#
# PRODUCTS 
#
def create_product(**kwargs):
    ident=kwargs.pop('id')
    p=Product(id=ident)
    p=_update_object(p,kwargs)
    return p.save()


def add_bands(bands):
    [ add_band(**b) for b in bands ] 


def add_band(**kwargs):
    band_type=kwargs.pop('type',None)
    band_name=kwargs.pop('name')
    product_id=Product.namespace_id(kwargs.pop('product_id'))
    band=_dl_band(band_type)
    b=band(name=band_name,product_id=product_id)
    b=_update_object(b,kwargs)
    return b.save()


def delete_product(
        product_id,
        add_namespace=False,
        cascade=True,
        i_am_absolutely_sure=False):
    """ WARNING: THIS DELETES THE PRODUCT AND ALL IMAGERY
    """
    if i_am_absolutely_sure:
        resp=Catalog().remove_product(
            product_id, 
            add_namespace=add_namespace, 
            cascade=cascade )
        return resp
    else:
        print(f'DO YOU REALLY WANT TO DELETE {product_id}?')
        return False


def upload(
        im,
        product_id,
        image_id,
        date,
        rinfo,
        extra_properties={}):
    dl_img=Image(
        product_id=Product.namespace_id(product_id), 
        name=image_id)
    dl_img.acquired=date
    dl_img.extra_properties=extra_properties
    return dl_img.upload_ndarray(
        im,
        upload_options=None, 
        raster_meta=rinfo,
        overviews=OVERVIEWS,
        overview_resampler=OVERVIEW_RESAMPLER)


#
# INTERNAL
#
def _is_str(value):
    """ is_str method that works for py2 and py3 """
    if isinstance(value,str):
        return True
    else:
        try: 
            is_a_str=isinstance(out,unicode)
        except:
            is_a_str=False
        return is_a_str


def _coordinate_info(aoi):
    sz=aoi.tilesize + (2 * aoi.pad)
    meta={
        'coordinateSystem': {
            'proj4': aoi.proj4,
            'wkt': aoi.wkt
        },
        'driverLongName': 'In Memory Raster',
        'driverShortName': 'MEM',
        'geoTransform': aoi.geotrans,
        'metadata': {'': {'Corder': 'RPCL', 'id': '*'}},
        'size': [sz, sz],
        'files': []
    }
    return meta


def _dl_band(band_type):
    if band_type==CLASS:
        return ClassBand
    elif band_type==SPECTRAL:
        return SpectralBand
    elif band_type==MICROWAVE:
        return MicrowaveBand
    elif band_type==MASK:
        return MaskBand
    else:
        return GenericBand


def _update_object(obj,data):
    for key,value in data.items():
        setattr(obj,key,value)
    return obj


