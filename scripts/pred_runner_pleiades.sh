#
#
# NOTE: FIRST UPDATE HARD CODED predict/gee_ingest py files
#
#



#
# RUN
#
echo
echo
echo gee_ingest_pleiades
date
echo 
python gee_ingest_pleiades.py
echo
echo


echo
echo
echo gee_ingest_naip
date
echo 
python gee_ingest_naip.py
echo
echo


echo
echo
echo gee_ingest_pleiades_weights
date
echo 
python gee_ingest_pleiades_weights.py
echo
echo


echo
echo
echo gee_ingest_pleiades_gbu
date
echo 
python gee_ingest_pleiades_gbu.py
echo
echo



