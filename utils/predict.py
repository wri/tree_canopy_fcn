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
        dates,
        bands,
        extra_properties={}):
    def _upload(args):
        key,date,preds,cats=args
        rinfo=dlabs.raster_info(key,bands)
        image_id=h.get_image_id(key,date)
        im=np.stack([cats,preds])
        print(image_id,im.shape)
        task=dlabs.upload(
                im,
                product_id,
                image_id,
                date,
                rinfo,
                extra_properties=extra_properties)
        return image_id, task
    _,preds,cats=batch(model,batch_keys)
    args_list=list(zip(batch_keys,dates,preds,cats))
    return mproc.map_with_threadpool(_upload,args_list,len(args_list))








