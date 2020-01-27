import re
import numpy as np
import torch
import mproc
import torch_kit.helpers as H
import utils.helpers as h
import utils.load as load
import utils.dlabs as dlabs
from config import SAVE_LOCAL


#
# CONSTANTS
#
THRESHOLD=0.25




#
# PREDICT
#
def batch(model,batch_keys):
    input_batch=torch.Tensor(load.batch(batch_keys))
    if torch.cuda.is_available():
        input_batch=input_batch.cuda()
    preds=model(input_batch).squeeze(dim=1)
    cats=(preds>THRESHOLD)
    return H.to_numpy(input_batch), H.to_numpy(preds), H.to_numpy(cats).astype(np.uint8)


def descartes_run(
        product_id,
        model,
        batch_keys,
        date,
        bands,
        extra_properties={}):
    def _upload(args):
        props=extra_properties.copy()
        tile_key,date,pred,cat=args
        hist=h.get_histogram(cat)
        pct=hist['Tree']/(hist['Tree']+hist['Not Tree'])
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
    _,preds,cats=batch(model,batch_keys)
    if not isinstance(date,list):
        date=[date]*len(batch_keys)
    args_list=list(zip(batch_keys,date,preds,cats))
    return mproc.map_with_threadpool(_upload,args_list,len(args_list))








