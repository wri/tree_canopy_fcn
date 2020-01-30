from random import shuffle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from image_kit.handler import InputTargetHandler
from config import HEIGHT_THRESHOLD, NDVI_THRESHOLD
#
# CONFIG
#
INPUT_DTYPE=np.float
TARGET_DTYPE=np.float
DEFAULT_BATCH_SIZE=8



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
            shuffle_data=shuffle):
        self.train_mode=train_mode
        self.handler=InputTargetHandler(
            means=means,
            stdevs=stdevs,
            band_indices=band_indices,
            augment=augment,
            target_squeeze=False,
            input_dtype=INPUT_DTYPE,
            target_dtype=TARGET_DTYPE)
        self.dataframe=dataframe
        if shuffle:
            self.dataframe=self.dataframe.sample(frac=1)


    def __len__(self):
        return len(self.dataframe)         
    

    def __getitem__(self, index):
        self.select_data(index)
        self.handler.set_augmentation()
        if self.train_mode:
            inpt=self.handler.input(self.input_path,return_profile=False)
            targ=self.handler.target(self.target_path,return_profile=False)
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
    def _postprocess_target(self,targ,ht_thresh=HEIGHT_THRESHOLD,ndvi_thresh=NDVI_THRESHOLD):
        targ = np.nan_to_num(targ)
        targ = ((targ[0,:,:]>ht_thresh) & (targ[1,:,:]>ndvi_thresh)).astype(np.float)       
        return np.expand_dims(targ,axis=0)


    def _safe_value(self,k,value):
        if isinstance(value,(int,float,str)):
            return value
        else:
            return str(value)

            
    def _clean(self,obj):
        return { k: self._safe_value(k,v) for k,v in obj.items() if self._safe_value(k,v) is not None }
