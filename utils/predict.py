import re
import numpy as np
import torch
import mproc
import torch_kit.helpers as H
import utils.helpers as h
import utils.load as load
import utils.dlabs as dlabs
from config import LOCAL_SRC


#
# CONSTANTS
#
THRESHOLD=0.25
NO_DATA_VALUE=2



#
# PREDICT
#
def batch(
        model,
        batch_keys,
        year=None,
        start=None,
        end=None,
        local_src=LOCAL_SRC):
    if local_src:
        input_batch,nodatas=load.batch(batch_keys)
    else:
        input_batch,nodatas=load.dl_batch(batch_keys,year=year,start=start,end=end)
    if input_batch is False:
        print('EMPTY:',batch_keys)
        return False, False, False
    else:
        input_batch=torch.Tensor(input_batch)
        if torch.cuda.is_available():
            input_batch=input_batch.cuda()
        preds=model(input_batch).squeeze(dim=1)
        cats=(preds>THRESHOLD)
        cats=H.to_numpy(cats).astype(np.uint8)
        cats[nodatas]=NO_DATA_VALUE
        return H.to_numpy(input_batch), H.to_numpy(preds), cats


def descartes_run(
        product_id,
        model,
        batch_keys,
        date,
        bands,
        extra_properties={},
        year=None,
        start=None,
        end=None,
        local_src=LOCAL_SRC):
    def _upload(args):
        tile_key,date,pred,cat=args
        props=extra_properties.copy()
        hist=h.get_histogram(cat)
        nb_valid_pix=(hist['Tree']+hist['Not Tree'])
        if nb_valid_pix:
            pct=hist['Tree']/nb_valid_pix
        else:
            pct='NaN'
        props['tile_key']=tile_key
        props['hist']=str(hist)
        props['percent_tree']=pct
        props['year']=str(date).split('-')[0][:4]
        rinfo=dlabs.raster_info(tile_key,bands)
        image_id=h.get_image_id(tile_key,date)
        im=np.stack([cat,pred])
        try:
            task=dlabs.upload(
                    im,
                    product_id,
                    image_id,
                    date,
                    rinfo,
                    extra_properties=props)
            return image_id, task
        except Exception as e:
            return image_id, str(e)
    _,preds,cats=batch(
        model,
        batch_keys,
        year=year,
        start=start,
        end=end,
        local_src=local_src)
    if preds is False:
        return []
    else:
        if not isinstance(date,list):
            date=[date]*len(batch_keys)
        args_list=list(zip(batch_keys,date,preds,cats))
        return mproc.map_with_threadpool(_upload,args_list,len(args_list))








