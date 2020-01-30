import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplib_colors
import matplotlib.ticker as mplib_ticker
import rasterio.plot as rplt
import seaborn as sns
import image_kit.processor as proc
from config import MEANS, STDEVS


VALUE_MAP_COLORS={
    'NotTree':"#ffffff",
    'Tree': "#4daf4a"
}

#
# HELPERS
#
def line(char='-',length=75,newlines=0):
    vspace(newlines)
    print(char*length)


def vspace(nb_lines=1):
    for _ in range(nb_lines): 
        print("\n")


#
# PLT
#
def catshow(im,ax=None,figsize=(6,4),color_dict='aue',alpha=1,color_bar=True):
    if color_dict=='aue':
        color_dict=AUE_COLORS
    elif color_dict=='bup':
        color_dict=VALUE_MAP_BUILT_UP_COLORS
    elif color_dict=='vmap':
        color_dict=VALUE_MAP_COLORS
    nb_cats=len(color_dict.keys())-1
    if im.ndim==3:
#         im=proc.to_bands_last(im)
#         im=im[:,:,0]
        im=im[0,:,:]
    cmap=mplib_colors.ListedColormap(color_dict.values())
    if not ax:
        fig,ax=plt.subplots(1,figsize=figsize)
    showax=ax.imshow(
            im,
            vmin=0,
            vmax=nb_cats,
            cmap=cmap,
            alpha=alpha)
    if color_bar:
        categories=color_dict.keys()
        cbar=plt.colorbar(showax,ax=ax)
        cbar.ax.yaxis.set_major_locator(mplib_ticker.LinearLocator(
            numticks=len(categories)))
        cbar.ax.yaxis.set_ticklabels(categories)
    return ax


def display_input_target(inpt,targ,alpha=0.6,figsize=(18,8),color_bar=False,color_dict='vmap'):
    inpt=proc.to_bands_last(inpt)
    if inpt.ndim>2:
        if inpt.shape[-1]>3:
            inpt=inpt[:,:,:3]
    if (targ.ndim>2) and (targ.shape[-1]>1):
        targ=targ.argmax(axis=-1)
    fig,axs=plt.subplots(1,2,figsize=figsize)
    inpt=inpt.astype(np.uint8)
    targ=targ.astype(np.uint8)
    axs[0].imshow(inpt)
    axs[1].imshow(inpt)
    axs[1].set_axis_off()
    catshow(targ,color_dict=color_dict,ax=axs[1],color_bar=color_bar,alpha=alpha)
    plt.show()


def display_input_target_prediction(
        inpt,
        targ,
        pred,
        alpha=0.9, #0.7,
        figsize=(18,6),
        color_bar=False,
        color_dict=None):
#         color_dict='vmap'):
#     inpt=proc.to_bands_last(inpt)
#     pred=proc.to_bands_last(pred)
#     if inpt.shape[-1]>3:
#         inpt=inpt[:,:,:3]
    if inpt.shape[0]>3:
        inpt=inpt[3,:,:]
#     if (targ.ndim>2) and (targ.shape[-1]>1):
#         targ=targ.argmax(axis=-1)
    fig,axs=plt.subplots(1,3,figsize=figsize)
#     inpt=inpt.astype(np.uint8)
#     targ=targ.astype(np.uint8)
    axs[0].imshow(inpt)
    axs[1].imshow(inpt)
    axs[2].imshow(inpt)
    axs[1].set_axis_off()
    axs[2].set_axis_off()
    catshow(
        targ,
        color_dict=color_dict,
        ax=axs[1],
        color_bar=False,
        alpha=alpha)
    catshow(
        pred,
        color_dict=color_dict,
        ax=axs[2],
        color_bar=color_bar,
        alpha=alpha)
    axs[0].set_title('INPUT')
    axs[1].set_title('TARGET')
    axs[2].set_title('PREDICITON')
    plt.show()
