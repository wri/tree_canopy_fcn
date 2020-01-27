import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplib_colors
import matplotlib.ticker as mplib_ticker
import rasterio.plot as rplt
import seaborn as sns
import image_kit.processor as proc




def show_results(im,pred,cat):
    fig,axs=plt.subplots(1,3,figsize=[9,3])
    rplt.show(proc.denormalize(im[:4],means=MEANS,stdevs=STDEVS),ax=axs[0])
    rplt.show(pred,ax=axs[1])
    rplt.show(cat,ax=axs[2])
    plt.show()