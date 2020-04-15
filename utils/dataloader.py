from random import shuffle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from image_kit.handler import InputTargetHandler
from config import HEIGHT_THRESHOLD, NDVI_THRESHOLD, GREENSPACE_THRESHOLDS
from config import BUILTUP_CATEGORY_THRESHOLDS

#
# CONFIG
#
INPUT_DTYPE=np.float
TARGET_DTYPE=np.float
DEFAULT_BATCH_SIZE=8



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
