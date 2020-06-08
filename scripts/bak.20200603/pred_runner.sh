#
#
# NOTE: FIRST UPDATE HARD CODED predict/gee_ingest py files
#
#


# 
# CONFIG
#
REGION=belohorizonte
DATA_DIR=/DATA/predictions
SCRIPT_DIR=/home/ericp/tree_canopy_fcn/repo/scripts
GCS_FOLDER=lidar-lulc/GREENSPACE/predictions
INPUT_TYPE=pleiades
PV=naip-v1


#
# RUN
# 
echo
echo
echo predict
date
echo 
python predict.py
echo
echo


echo
echo
echo upload to gcs: gs://$GCS_FOLDER/$REGION/$PV/
date
echo 
gsutil -m cp $DATA_DIR/$PV/$REGION/$INPUT_TYPE/*.tif gs://$GCS_FOLDER/$REGION/$PV/
echo
echo 


echo
echo
echo collect uris: $SCRIPT_DIR/$REGION-$PV-uris.csv
date
echo 
echo uri > $SCRIPT_DIR/$REGION-$PV-uris.csv
gsutil ls gs://$GCS_FOLDER/$REGION/$PV/*.tif >> $SCRIPT_DIR/$REGION-$PV-uris.csv
echo
echo


echo
echo
echo gee_ingest
date
echo 
python gee_ingest.py
echo
echo
