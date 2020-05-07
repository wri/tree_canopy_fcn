from random import shuffle
from copy import deepcopy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from image_kit.handler import InputTargetHandler
import image_kit.indices as indices
from config import HEIGHT_THRESHOLD, NDVI_THRESHOLD, GREENSPACE_THRESHOLDS
from config import BUILTUP_CATEGORY_THRESHOLDS


#
# CONFIG
#
INPUT_DTYPE=np.float
TARGET_DTYPE=np.float
DEFAULT_BATCH_SIZE=8
HAG_MIN=-3
HAG_MIN_VALUE=0
CATEGORY_BOUNDS=[   
    {
        'category': 'water',
        # 'value': 1,
        'hex': '#56CEFB',
        'ndwi': 0.35,
    },
    {
        'category': 'grass',
        # 'value': 2,
        'hex':'#ffff00',
        'ndvi': 0.2,
        'height': {'max': 0.5}
    },
    {
        'category': 'shrub',
        # 'value': 3,
        'hex':'#cc9900',
        'ndvi': 0.2,
        'height': [0.5,2]
    },
    {
        'category': 'tree',
        # 'value': 4,
        'hex':'#00ff00',
        'ndvi': 0.25,
        'height': [2,4]
    },
    {
        'category': 'big-tree',
        # 'value': 5,
        'hex': '#006400',
        'ndvi': 0.25,
        'height': 4
    },
    {
        'category': 'road',
        # 'value': 6,
        'hex': '#ffffff',
        'ndvi': {'max': 0},
        'height': {'max': 1.6}
    },
    {
        'category': '1-story',
        # 'value': 7,
        'hex': '#6600ff',
        'ndvi': {'max': 0},
        'height': [1.6,5]
    },
    {
        'category': '2to3-story',
        # 'value': 8,
        'hex': '#ff0000',
        'ndvi': {'max': 0},
        'height': [5,10]
    },
    {
        'category': '4+-story',
        # 'value': 9,
        'hex': '#ff00ff',
        'ndvi': {'max': 0},
        'height': 10
    }
]

# def stretch(im,min_value=-1,max_value=1):
#     mn=im.min()
#     mx=im.max()
#     return ((max_value-min_value)*(im-mn)/(mx-mn))+min_value


# def max_shift(im,min_value=0.8):
#     mx=im.max()
#     if mx<min_value:
#         im=im+(min_value-mx)
#     return im


# def min_shift(im,max_value=-0.6):
#     mn=im.min()
#     if mn>max_value:
#         im=im-(mn-max_value)
#     return im

# def dstd(im):
#     return im/im.std()




#
#
#
#  I SHOULD CALC INDICES FOR INPUT USUALLY!!!!
#
#
#
AB_MEANS=[100.83741572079242, 100.4938850966076, 86.63500986931308, 118.72746674454453]
AB_STDEVS=[42.098045003124774, 39.07388735786421, 39.629813116928815, 34.72351480486876]

class HeightIndexDataset(Dataset):
    """dataset from height and spectral indices"""
    @classmethod
    def loader(cls,
            dataframe,
            batch_size=DEFAULT_BATCH_SIZE,
            partial_batches=False,
            train_mode=True,
            hag_property=False,
            shuffle_data=True,
            loader_kwargs={},
            **kwargs):
        r""" convenience method for loading the DataLoader directly.
            Args:
                see class args

            Returns:
                dataloader 
        """
        return DataLoader(
                cls(dataframe,shuffle_data=shuffle_data,**kwargs),
                batch_size=batch_size,
                shuffle=shuffle_data,
                **loader_kwargs)        
    

    def __init__(self,
            dataframe,     
            means=None,
            stdevs=None,
            augment=False,
            band_indices=['ndvi','ndwi'],
            center_indices=False,
            category_bounds=CATEGORY_BOUNDS,
            input_band_count=4,
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE,
            hag_min=HAG_MIN,
            hag_min_value=HAG_MIN_VALUE,
            cropping=None,
            float_cropping=None,
            width=None,
            height=None,
            example_path=None,
            no_data_value=0,
            train_mode=False,
            hag_property=True,
            shuffle_data=False):
        self.train_mode=train_mode
        self.handler=InputTargetHandler(
            means=means,
            stdevs=stdevs,
            band_indices=band_indices,
            augment=augment,
            cropping=cropping,
            float_cropping=float_cropping,
            width=width,
            height=height,
            example_path=example_path,
            target_squeeze=False,
            input_dtype=input_dtype,
            target_dtype=np.float)
        self._set_spectral_bands(band_indices,input_band_count)
        self._set_hag_properties(hag_min,hag_min_value,hag_property)
        self.center_indices=center_indices
        self.no_data_value=no_data_value
        self.target_dtype=target_dtype
        self.category_bounds=category_bounds
        self.dataframe=dataframe
        if shuffle_data:
            self.dataframe=self.dataframe.sample(frac=1)


    def __len__(self):
        return len(self.dataframe)         
    

    def __getitem__(self, index):
        self.select_data(index)
        self.handler.set_window()
        self.handler.set_augmentation()
        if self.train_mode:
            inpt=self.handler.input(self.rgbn_path,return_profile=False)
            hag=self._load_hag(self.hag_path,return_profile=False)
            targ=self._build_target(inpt,hag)
            return {
                'input': inpt, 
                'target': targ }
        else:
            inpt,inpt_p=self.handler.input(self.rgbn_path,return_profile=True)
            hag,hag_p=self._load_hag(self.hag_path,return_profile=True)
            targ=self._build_target(inpt,hag)
            self.input=inpt
            self.targ=targ
            self.ndvi=self._get_spectral_band(inpt,'ndvi')
            self.ndwi=self._get_spectral_band(inpt,'ndwi')
            row=self._clean(self.row.to_dict())
            inpt_p=self._clean(inpt_p)
            hag_p=self._clean(hag_p)
            itm={
                'input': inpt, 
                'target': targ,
                'index': self.index,
                'rgbn_path': self.rgbn_path,
                'hag_path': self.hag_path,
                'k': self.handler.k,
                'flip': self.handler.flip,
                'row': row,
                'rbgn_profile': inpt_p,
                'hag_profile': hag_p 
                }
        return itm


    def select_data(self,index):
        self.index=index
        self.row=self.dataframe.iloc[index]
        self.rgbn_path=self.row.rgbn_path
        self.hag_path=self.row.hag_path


    #
    # INTERNAL METHODS
    #
    def _set_spectral_bands(self,band_indices,band_count):
        if band_indices:
            self.spectral_bands={ b:(band_count+i) for i,b in enumerate(band_indices) }
        else:
            self.spectral_bands={}


    def _get_spectral_band(self,inpt,index_name):
        band_index=self.spectral_bands.get(index_name,False)
        if band_index is False:
            im = indices.index(inpt,index_name)
        else:
            im = inpt[band_index]
        if self.center_indices:
            return (im-im.mean())#/im.std()
        else:
            return im


    def _set_hag_properties(self,hag_min,hag_min_value,hag_property):
        self.hag_min=hag_min
        if hag_min_value is None:
            self.hag_min_value=hag_min
        else:
            self.hag_min_value=hag_min_value
        self.hag_property=hag_property

        
    def _load_hag(self,path,return_profile=False):
        hag,p=self.handler.target(path,return_profile=True)
        hag=hag[0]
        if self.hag_min is not None:
            hag[hag<self.hag_min]=self.hag_min_value
        if self.hag_property:
            self.hag=hag
        if return_profile:
            return hag, p
        else:
            return hag

            
    def _build_target(self,rgbn,hag):
        cat=np.full_like(hag,self.no_data_value)
        hag=np.nan_to_num(hag)
        for i,bnds in enumerate(self.category_bounds):
            cat[self._is_category(rgbn,hag,bnds)]=bnds.get('value',i)
        return cat.astype(self.target_dtype)
    
    
    def _is_category(self,rgbn,hag,bounds):
        ndvi_b, ndvi=self._process_spectral_index(rgbn,'ndvi',bounds)
        ndwi_b, ndwi=self._process_spectral_index(rgbn,'ndwi',bounds)
        height_b=self._process_bounds(hag,bounds.get('height'))
        or_bnds=bounds.get('or')
        if or_bnds:
            or_ndvi_b, ndvi=self._process_spectral_index(rgbn,'ndvi',or_bnds,ndvi,0)
            or_ndwi_b, ndwi=self._process_spectral_index(rgbn,'ndwi',or_bnds,ndwi,0)
            or_height_b=self._process_bounds(hag,bounds.get('height'),0)
            or_b=(or_ndwi_b+or_ndvi_b+or_height_b).astype(bool) 
        else:
            or_b=1
        return (ndwi_b*ndvi_b*height_b*or_b).astype(bool) 


    def _process_spectral_index(self,inpt,index_name,bounds,index_im=None,pass_value=1):
        bnds=bounds.get(index_name)
        if bnds:
            if index_im is None:
                index_im=self._get_spectral_band(inpt,index_name)
            im_b=self._process_bounds(index_im,bnds,pass_value)
        else:
            im_b=pass_value
            index_im=None
        return im_b, index_im


    def _process_bounds(self,data,bounds,pass_value=1):
        if bounds is not None:
            if isinstance(bounds,list):
                data=(data>=bounds[0])*(data<bounds[1])
            elif isinstance(bounds,dict):
                mx=bounds.get('max')
                mn=bounds.get('min')
                if (mx is not None) and (mn is not None):
                    data=(data<mx)*(data>=mn)
                elif (mx is not None):
                    data=(data<mx)
                elif (mn is not None):
                    data=(data>=mn)
            else:
                data=(data>=bounds)
        else:
            data=pass_value
        return data
        
    
    def _safe_value(self,k,value):
        if isinstance(value,(int,float,str)):
            return value
        else:
            return str(value)

            
    def _clean(self,obj):
        return { k: self._safe_value(k,v) for k,v in obj.items() if self._safe_value(k,v) is not None }






















###############################################################################
###############################################################################
#
#
# OLD DATASETS
#
#
###############################################################################
###############################################################################






class HeightGreenDataset(Dataset):
    """Urban Tree Canopy dataset."""
    
    @classmethod
    def loader(cls,
            dataframe,
            batch_size=DEFAULT_BATCH_SIZE,
            partial_batches=False,
            loader_kwargs={},
            shuffle_data=False,
            **kwargs):
        r""" convenience method for loading the DataLoader directly.
            Args:
                see class args

            Returns:
                dataloader 
        """
        return DataLoader(
                cls(dataframe,shuffle_data=shuffle_data,**kwargs),
                batch_size=batch_size,
                shuffle=shuffle_data,
                **loader_kwargs)        
    

    def __init__(self,
            dataframe,     
            means=None,
            stdevs=None,
            band_indices=['ndvi'],
            augment=False,
            train_mode=True,
            category_thresholds=BUILTUP_CATEGORY_THRESHOLDS,
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE,
            shuffle_data=shuffle):
        self.train_mode=train_mode
        self.handler=InputTargetHandler(
            means=means,
            stdevs=stdevs,
            band_indices=band_indices,
            augment=augment,
            target_squeeze=False,
            input_dtype=input_dtype,
            target_dtype=np.float)
        self.target_dtype=target_dtype
        self.category_thresholds=category_thresholds
        self.dataframe=dataframe
        if shuffle_data:
            self.dataframe=self.dataframe.sample(frac=1)


    def __len__(self):
        return len(self.dataframe)         
    

    def __getitem__(self, index):
        self.select_data(index)
        self.handler.set_augmentation()
        if self.train_mode:
            inpt=self.handler.input(self.input_path,return_profile=False)
            targ=self.handler.target(self.target_path,return_profile=False)
            targ=self._postprocess_target(targ,inpt)
            return {
                'input': inpt, 
                'target': targ }
        else:
            inpt,inpt_p=self.handler.input(self.input_path,return_profile=True)
            targ,targ_p=self.handler.target(self.target_path,return_profile=True)
            targ=self._postprocess_target(targ,inpt)
            row=self._clean(self.row.to_dict())
            inpt_p=self._clean(inpt_p)
            targ_p=self._clean(targ_p)
            itm={
                'input': inpt, 
                'target': targ,
                'index': self.index,
                'input_path': self.input_path,
                'target_path': self.target_path,
                'k': self.handler.k,
                'flip': self.handler.flip,
                'row': row,
                'input_profile': inpt_p,
                'target_profile': targ_p 
                }
        return itm


    def select_data(self,index):
        self.index=index
        self.row=self.dataframe.iloc[index]
        self.input_path=self.row.imagery_path
        self.target_path=self.row.raster_path


    #
    # INTERNAL METHODS
    #
    def _get_height_ranges(self,thresholds,min_value=0):
        ranges=[[min_value,thresholds[0]]]
        for b in range(0,len(thresholds)-1):
                ranges.append([thresholds[b],thresholds[b+1]])
        ranges.append([thresholds[-1],None])
        return ranges


    def _is_category(self,targ,ndwi,cat_map):
        targ=np.nan_to_num(targ)
        bnd=cat_map.get('ndwi')
        if bnd is None:
            bnd=cat_map.get('ndvi')
            im=self._within_bound(targ[1],bnd,cat_map.get('max_bound'))
        else:
            im=self._within_bound(ndwi,bnd,cat_map.get('max_bound'))
        bnd=cat_map.get('height')
        if bnd is not None:
            h=self._within_bound(targ[0],bnd,False)
            im=(im*h)
        return im.astype(bool)


    def _within_bound(self,im,bnd,max_bound):
        if isinstance(bnd,list):
            im=(im<bnd[1])*(im>=bnd[0])
        else:
            if max_bound:
                im=(im<bnd)
            else:
                im=(im>=bnd)
        return im


    def _postprocess_target(self,targ,inpt):
        cat=np.full_like(targ[0],0)
        targ=np.nan_to_num(targ)
        for i,cmap in enumerate(self.category_thresholds,start=1):
            cat[self._is_category(targ,inpt[5],cmap)]=cmap.get('value',i)
        return cat.astype(self.target_dtype)
         

    def _safe_value(self,k,value):
        if isinstance(value,(int,float,str)):
            return value
        else:
            return str(value)

            
    def _clean(self,obj):
        return { k: self._safe_value(k,v) for k,v in obj.items() if self._safe_value(k,v) is not None }







class UrbanTreeDataset(Dataset):
    """Urban Tree Canopy dataset."""
    
    @classmethod
    def loader(cls,
            dataframe,
            batch_size=DEFAULT_BATCH_SIZE,
            partial_batches=False,
            loader_kwargs={},
            shuffle_data=False,
            **kwargs):
        r""" convenience method for loading the DataLoader directly.
            Args:
                see class args

            Returns:
                dataloader 
        """
        return DataLoader(
                cls(dataframe,shuffle_data=shuffle_data,**kwargs),
                batch_size=batch_size,
                shuffle=shuffle_data,
                **loader_kwargs)        
    

    def __init__(self,
            dataframe,     
            means=None,
            stdevs=None,
            band_indices=['ndvi'],
            augment=False,
            train_mode=True,
            height_thresholds=HEIGHT_THRESHOLD,
            ndvi_threshold=NDVI_THRESHOLD,
            greenspace=True,
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE,
            shuffle_data=shuffle):
        self.train_mode=train_mode
        self.handler=InputTargetHandler(
            means=means,
            stdevs=stdevs,
            band_indices=band_indices,
            augment=augment,
            target_squeeze=False,
            input_dtype=input_dtype,
            target_dtype=np.float)
        self.target_dtype=target_dtype
        self.greenspace=greenspace
        self._set_height_threshods(height_thresholds)
        self.ndvi_threshold=ndvi_threshold
        self.dataframe=dataframe
        if shuffle_data:
            self.dataframe=self.dataframe.sample(frac=1)


    def _set_height_threshods(self,height_thresholds):
        self.multi_threshold=isinstance(height_thresholds,list)
        self.height_thresholds=height_thresholds
        if self.multi_threshold:
            self.height_ranges=self._get_height_ranges(height_thresholds)


    def __len__(self):
        return len(self.dataframe)         
    

    def __getitem__(self, index):
        self.select_data(index)
        self.handler.set_augmentation()
        if self.train_mode:
            inpt=self.handler.input(self.input_path,return_profile=False)
            targ=self.handler.target(self.target_path,return_profile=False)
            if self.multi_threshold:
                targ=self._postprocess_target(targ)
            else:
                targ=self._postprocess_target(targ)
            return {
                'input': inpt, 
                'target': targ }
        else:
            inpt,inpt_p=self.handler.input(self.input_path,return_profile=True)
            targ,targ_p=self.handler.target(self.target_path,return_profile=True)
            targ=self._postprocess_target(targ)
            row=self._clean(self.row.to_dict())
            inpt_p=self._clean(inpt_p)
            targ_p=self._clean(targ_p)
            itm={
                'input': inpt, 
                'target': targ,
                'index': self.index,
                'input_path': self.input_path,
                'target_path': self.target_path,
                'k': self.handler.k,
                'flip': self.handler.flip,
                'row': row,
                'input_profile': inpt_p,
                'target_profile': targ_p 
                }
        return itm


    def select_data(self,index):
        self.index=index
        self.row=self.dataframe.iloc[index]
        self.input_path=self.row.imagery_path
        self.target_path=self.row.raster_path


    #
    # INTERNAL METHODS
    #
    def _get_height_ranges(self,thresholds,min_value=0):
        ranges=[[min_value,thresholds[0]]]
        for b in range(0,len(thresholds)-1):
                ranges.append([thresholds[b],thresholds[b+1]])
        ranges.append([thresholds[-1],None])
        return ranges


    def _postprocess_target(self,targ):
        targ=np.nan_to_num(targ)
        if self.greenspace:
            green=(targ[1]>=self.ndvi_threshold)
        else:
            green=(targ[1]<self.ndvi_threshold)
        if self.multi_threshold:
            height_cat=self._get_height_categories(targ[0])
        else:
            height_cat=(targ[0]>=self.height_thresholds)
        return (green*height_cat).astype(self.target_dtype)


    def _get_height_categories(self,height):
        hcat=np.full_like(height,0)
        for i,(mn,mx) in enumerate(self.height_ranges,start=1):
              hcat[self._height_test(height,mn,mx)]=i
        return hcat


    def _height_test(self,height,min_height,max_height):
        test=(height>=min_height)
        if max_height:
            test*=(height<max_height) 
        return test
         

    def _safe_value(self,k,value):
        if isinstance(value,(int,float,str)):
            return value
        else:
            return str(value)

            
    def _clean(self,obj):
        return { k: self._safe_value(k,v) for k,v in obj.items() if self._safe_value(k,v) is not None }
