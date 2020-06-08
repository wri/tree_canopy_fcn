from random import shuffle
from copy import deepcopy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from image_kit.handler import InputTargetHandler
import image_kit.indices as indices
import image_kit.processor as proc
import utils.load as load


#
# CONFIG
#
INPUT_DTYPE=np.float
TARGET_DTYPE=np.float
DEFAULT_BATCH_SIZE=8
HAG_MIN=-3
HAG_MIN_VALUE=0


#
# HELPERS
#
def get_category_bounds(bounds,*keys):
    bounds=load.bounds(bounds)
    catbnds=[]
    for k in keys:
        catbnds+=bounds[k]
    return catbnds 
 


class HeightIndexDataset(Dataset):
    NO_DATA_LAST='no_data_last'



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
            input_bands=None,
            band_indices=['ndvi','ndwi'],
            has_target_paths=False,
            value_map=None,
            target_rgbn=True,
            input_bounds=None,
            category_bounds=None,
            category_product='naip',
            category_keys=['grass','green'],
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
            no_data_value=NO_DATA_LAST,
            target_methods=None,
            uncertain_value=None,
            smoothing_kernel=None,
            nb_categories=None,
            train_mode=False,
            hag_property=True,
            shuffle_data=False):
        self.train_mode=train_mode
        self.target_rgbn=target_rgbn
        self.smoothing_kernel=smoothing_kernel
        self.nb_categories=nb_categories
        self.handler=InputTargetHandler(
            means=means,
            stdevs=stdevs,
            input_bands=input_bands,
            band_indices=band_indices,
            input_bounds=input_bounds,
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
        self.has_target_paths=has_target_paths
        self.value_map=value_map
        self.target_dtype=target_dtype
        if category_bounds:
            self.category_bounds=category_bounds
        else:
            self.category_bounds=get_category_bounds(category_product,*category_keys)
        self.target_methods=target_methods
        self.uncertain_value=uncertain_value
        if no_data_value==HeightIndexDataset.NO_DATA_LAST:
            self.no_data_value=len(self.category_bounds)
        elif no_data_value:
            self.no_data_value=no_data_value
        else:
            self.no_data_value=0
        self.dataframe=dataframe
        if shuffle_data:
            self.dataframe=self.dataframe.sample(frac=1)


    def input_data(self):
        return self.handler.input(self.input_path,return_profile=True)


    def rgbn_data(self,inpt=None,inpt_profile=None):
        if self.target_rgbn:
            rgbn, rgbn_p=self.handler.input(self.rgbn_path,return_profile=True)
        else:
            if inpt is None:
                rgbn, rgbn_p=self.input_data()
            else:
                rgbn, rgbn_p=inpt, inpt_profile
        return rgbn, rgbn_p


    def hag_data(self):
        return self._load_hag(self.hag_path,return_profile=True)


    def target_data(self,
            rgbn=None,
            hag=None,
            inpt=None,
            rgbn_profile=None,
            inpt_profile=None,
            return_profile=True):
        if self.has_target_paths:
            targ, targ_p=self.handler.target(self.target_path,return_profile=True)
            targ=targ[0]
        else:
            if rgbn is None:
                rgbn, rgbn_profile=self.rgbn_data(inpt,inpt_profile)
            if hag is None:
                hag, hag_p=self.hag_data()
            if return_profile:
                targ_p=rgbn_profile.copy()
                targ_p['count']=1
            targ=self._build_target(rgbn,hag)
        if self.value_map:
            targ=proc.map_values(targ,self.value_map)
        if self.smoothing_kernel is not None:
            targ=proc.categorical_smoothing(
                targ.astype(np.uint8),
                self.nb_categories,
                kernel=self.smoothing_kernel)
        targ=targ.astype(self.target_dtype)
        if return_profile:
            return targ, targ_p
        else:
            return targ


    def __len__(self):
        return len(self.dataframe)         
    

    def __getitem__(self, index):
        self.select_data(index)
        self.handler.set_window()
        self.handler.set_augmentation()
        inpt,inpt_p=self.input_data()
        if self.has_target_paths:
            targ=self.target_data(return_profile=False)
        else:
            rgbn, rgbn_p=self.rgbn_data(inpt, inpt_p)
            hag, hag_p=self.hag_data()
            targ=self.target_data(rgbn,hag,return_profile=False)
        if self.train_mode:
            return {
                'input': inpt, 
                'target': targ }
        else:
            row=self._clean(self.row.to_dict())
            inpt_p=self._clean(inpt_p)
            itm={
                'input': inpt, 
                'target': targ,
                'index': self.index,
                'row': row,
                'input_path': self.input_path,
                'k': self.handler.k,
                'flip': self.handler.flip,
                'input_profile': inpt_p,
            }
            if self.has_target_paths:
                itm['target_path']=self.target_path
            else:
                hag_p=self._clean(hag_p)
                rgbn_p=self._clean(rgbn_p)
                itm['rgbn']=rgbn
                itm['rgbn_path']=self.rgbn_path
                itm['rgbn_profile']=rgbn_p
                itm['hag']=hag
                itm['hag_path']=self.hag_path
                itm['hag_profile']=hag_p 

        return itm


    def select_data(self,index):
        self.index=index
        self.row=self.dataframe.iloc[index]
        self.input_path=self.row.input_path
        if self.has_target_paths:
            self.target_path=self.row.target_path
        else:
            self.hag_path=self.row.hag_path
            if self.target_rgbn:
                self.rgbn_path=self.row.rgbn_path
            else:
                self.rgbn_path=self.input_path


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
        if self.uncertain_value is not None:
            uncertian=np.full_like(hag,0)
        hag=np.nan_to_num(hag)

        for i,bnds in enumerate(self.category_bounds):
            test=self._is_category(rgbn,hag,bnds)
            cat[test]=bnds.get('value',i)
            if self.uncertain_value is not None:
                uncertian+=test
        if self.uncertain_value is not None:
            cat[uncertian>1]=self.uncertain_value
        if self.target_methods:
            for tm in self.target_methods:
                cat[tm['method'](rgbn,hag)]=tm['value']
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











