from random import shuffle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from image_kit.handler import InputTargetHandler
# import utils.helpers as h
# from config import DATA, RESOLUTION, VALUE_MAP, NB_CATEGORIES, FILL_ROADS_VALUE_MAP, S2_BANDS
#
# CONFIG
#
INPUT_DTYPE=np.float
TARGET_DTYPE=np.int  #Changed
DEFAULT_BATCH_SIZE=8
# DEFAULT S2 CONFIG
# S2_INDICIES=['ndvi','ndwi','ndbi','built_up']
# S2_STATS=h.get_aue_data('stats','master_all',typ='s2')


# S2_CONFIG={
#     'input_bands': None,
#     'band_indices': S2_INDICIES,
#     'means': S2_STATS['means'],
#     'stdevs': S2_STATS['stdevs'],
#     'cropping': False,
#     'float_cropping': 18,
#     'resolution': 5
# }
# DEFAULT AIRBUS
# AIRBUS_CONFIG={}

# Setting the threshold for height and ndvi
height_threshold = 4
ndvi_threshold = 0.1

# create a dataset class for our Urban Tree Canopy dataset
# set band_indices to None because ndvi, height already calculated in raster_path

class UrbanTreeDataset(Dataset):
    """Urban Tree Canopy dataset."""
    
    @classmethod
    def loader(cls,
            dataframe,
            batch_size=DEFAULT_BATCH_SIZE,
            partial_batches=False,
            loader_kwargs={},
            **kwargs):
        r""" convenience method for loading the DataLoader directly.
            
            Args:
                see class args

            Returns:
                dataloader 
        """


        return DataLoader(cls(dataframe,**kwargs),batch_size=batch_size,**loader_kwargs)        
    def __init__(self,
            dataframe,     
            means=None,
            stdevs=None,
            band_indices=['ndvi'],
            augment=False,
            value_map=None, # Changed
            train_mode=True,
            target_expand_axis=None,
            UPDATE_VERSION=None,# Changed
            shuffle_data=shuffle                 
                ):
        self.handler=InputTargetHandler(
            means=means,
            stdevs=stdevs,
            band_indices=band_indices,
            augment=augment,
            target_squeeze=False,
            target_dtype = np.float
        )
        
        self.canopy_frame = dataframe
        
    def __len__(self):
        return len(self.canopy_frame)         
    
    def __getitem__(self, index):
        
        self.select_data(index)
        
        self.handler.set_augmentation()
        
        inpt=self.handler.input(self.input_path,return_profile=False)
        
        targ=self.handler.target(self.target_path,return_profile=False)
    
        targ=self._postprocess_target(targ)

        return {
            'input': inpt, 
            'target': targ }      
        
    def select_data(self, index):
        self.index=index
        self.row=self.canopy_frame.iloc[index]
        self.input_path=self.row.imagery_path
        self.target_path=self.row.raster_path
    
    def _postprocess_target(self,targ, ht_thresh = height_threshold, ndvi_thresh = ndvi_threshold):
        targ = np.nan_to_num(targ)
        targ = ((targ[0,:,:]>ht_thresh) & (targ[1,:,:]>ndvi_thresh)).astype(np.float)       
        return np.expand_dims(targ,axis=0)




