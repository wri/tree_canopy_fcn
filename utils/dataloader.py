from random import shuffle
from copy import deepcopy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from image_kit.handler import InputTargetHandler
import image_kit.indices as indices
from config import CATEGORY_BOUNDS, NAIP_WATER_CATEGORY_BOUNDS
from config import NAIP_GREEN_CATEGORY_BOUNDS, NAIP_BU_CATEGORY_BOUNDS


#
# CONFIG
#
INPUT_DTYPE=np.float
TARGET_DTYPE=np.float
DEFAULT_BATCH_SIZE=8
HAG_MIN=-3
HAG_MIN_VALUE=0




class HeightIndexDataset(Dataset):
    NAIP_GREEN='naip_green'
    NAIP_BU='naip_bu'
    NAIP_ALL='naip_all'


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
            input_bounds=None,
            center_indices=False,
            category_bounds='naip_green',
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
        self.center_indices=center_indices
        self.no_data_value=no_data_value
        self.target_dtype=target_dtype
        self.category_bounds=self._category_bounds(category_bounds)
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
    def _category_bounds(self,category_bounds):
        if isinstance(category_bounds,str):
            if category_bounds==HeightIndexDataset.NAIP_GREEN:
                category_bounds=NAIP_GREEN_CATEGORY_BOUNDS
            elif category_bounds==HeightIndexDataset.NAIP_BU
                category_bounds=NAIP_BU_CATEGORY_BOUNDS
            elif category_bounds==HeightIndexDataset.NAIP_ALL
                category_bounds=NAIP_WATER_CATEGORY_BOUNDS
                category_bounds+=NAIP_GREEN_CATEGORY_BOUNDS
                category_bounds+=NAIP_BU_CATEGORY_BOUNDS
            else:
                raise ValueError(f'{category_bounds} is not valid')
        return category_bounds


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










