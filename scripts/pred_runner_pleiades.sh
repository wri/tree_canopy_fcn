#
#
# NOTE: FIRST UPDATE HARD CODED predict/gee_ingest py files
#
#


# 
# CONFIG
#
REGION=costa_rica-san_jose
DATA_DIR=/DATA/predictions
SCRIPT_DIR=/home/ericp/tree_canopy_fcn/repo/scripts
GCS_FOLDER=lidar-rgbn/dev/gbu/predictions
INPUT_TYPE=pleiades
PNAME=pleiades-dlv3p-green_bu-ndvi-ndwi
PV=v2

#
# RUN
# 
echo
echo
echo predict
date
echo 
echo python predict_pleiades.py
python predict_pleiades.py
echo
echo


echo
echo
echo upload to gcs: gs://$GCS_FOLDER/$REGION/$PNAME/$PV
date
echo 
echo gsutil -m cp $DATA_DIR/$PV/$REGION/$INPUT_TYPE/*.tif gs://$GCS_FOLDER/$REGION/$PV/$PNAME
gsutil -m cp $DATA_DIR/$PV/$REGION/$INPUT_TYPE/*.tif gs://$GCS_FOLDER/$REGION/$PV/$PNAME
echo
echo 


echo
echo
echo collect uris: $SCRIPT_DIR/$REGION-$PV-$PNAME-uris.csv
date
echo 
echo uri > $SCRIPT_DIR/$REGION-$PV-$PNAME-uris.csv
echo gsutil ls gs://$GCS_FOLDER/$REGION/$PV/$PNAME/*.tif >> $SCRIPT_DIR/$REGION-$PV-$PNAME-uris.csv
gsutil ls gs://$GCS_FOLDER/$REGION/$PV/$PNAME/*.tif >> $SCRIPT_DIR/$REGION-$PV-$PNAME-uris.csv
echo
echo


echo
echo
echo gee_ingest
date
echo 
echo python gee_ingest_pleiades.py
python gee_ingest_pleiades.py
echo
echo
