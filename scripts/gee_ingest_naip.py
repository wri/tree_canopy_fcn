import os
import re
import pandas as pd
#
# CONFIG
#
CSV='belohorizonte-naip-v1-uris.csv'
# USER='projects/wri-datalab'
USER='users/brookwilliams'
IC=f'{USER}/lidar-rgbn/dev/green/belohorizonte-naip-v1'
NO_DATA=4


#
# DATA
#
df=pd.read_csv(f'/home/ericp/tree_canopy_fcn/repo/scripts/{CSV}')
df.shape[0]


#
# RUN
#
CMD_TMPL='earthengine upload image {} --asset_id {} --pyramiding_policy MODE --nodata_value {} --force'
def ee_safe_name(path):
    name=path.split('/')[-1]
    name=re.sub('.tif$','',name)
    name=re.sub(r':','c',name)
    name=re.sub(r'\.','d',name)
    return name


def upload_im(uri,dry_run=False):
    aid=f'{IC}/{ee_safe_name(uri)}'
    cmd=CMD_TMPL.format(uri,aid,NO_DATA)
    os.system(cmd)


uris=df.uri.tolist()
for i,uri in enumerate(uris):
    print(i,'-'*10)
    upload_im(uri)
