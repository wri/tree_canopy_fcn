import re
import numpy as np
from descarteslabs.catalog import Image, Product, OverviewResampler
import torch_kit.helpers as H



#
# CONSTANTS
#
THRESHOLD=0.25
OVERVIEWS=[2,4,6,8,10,12]
OVERVIEW_RESAMPLER=OverviewResampler.MODE




#
# PREDICT
#
def run_predictions(model,batch_keys):
    batch=load_batch(batch_keys)
    preds=model(torch.Tensor(batch).to(DEVICE)).squeeze(dim=1)
    cats=(preds>THRESHOLD)
    return batch, H.to_numpy(preds), H.to_numpy(cats).astype(np.uint8)


#
# SAVE
#
def run_batch(model,batch_keys):
    _,preds,cats=run_predictions(model,batch_keys)
    save_batch(batch_keys,preds,cats)

    
def save_batch(batch_keys,preds,cats):
    if SAVE_LOCAL:
        save_method=save_local
    else:
        save_method=upload_to_dl
    args_list=list(zip(batch_keys,preds,cats))
    return mproc.map_with_threadpool(save_method,args_list,len(args_list))


def save_local(args):
    key,preds,cats=args
    print(key,preds.shape,cats.shape)


def upload_to_dl(args):
    key,preds,cats=args
    print(key,preds.shape,cats.shape)
    rinfo=raster_info(key)
    image_id=re.sub(r'\.','_',f'IM_{key}')
    image_id=re.sub(r':','--',image_id)
    im=np.stack([cats,preds])
    print(image_id,im.shape)
    dl_img=Image(
        product_id=Product.namespace_id(PROD_ID), 
        name=image_id)
    dl_img.acquired=DATE
    dl_img.upload_ndarray(
        im,
        upload_options=None, 
        raster_meta=rinfo,
        overviews=OVERVIEWS,
        overview_resampler=OVERVIEW_RESAMPLER)