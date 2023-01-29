#!/bin/bash

folder_name=$1

if test $# -lt 1 ; then echo 'Provide a folder_name in logs' ; exit 1 ; fi

if [ ! -d "logs/$folder_name" ]; then
  echo "logs/$folder_name does not exist."
  exit 1
fi

echo "******* Search on BeamNG + migration on Udacity and all individuals migrated on Donkey *******"
python compute_mapelites_correlation.py \
      --folder logs/$folder_name \
      --filepaths merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search merged_mapelites_migration_donkey_udacity_search_mapelites_migration_donkey_beamng_search \
      --load-probability-map \
      --failure-probability

echo "******* Search on Udacity + migration on BeamNG and all individuals migrated on Donkey *******"
python compute_mapelites_correlation.py \
      --folder logs/$folder_name \
      --filepaths merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search merged_mapelites_migration_donkey_udacity_search_mapelites_migration_donkey_beamng_search \
      --load-probability-map \
      --failure-probability

echo "******* Search on both BeamNG and Udacity (PROD) and all individuals migrated on Donkey *******"
python compute_mapelites_correlation.py \
      --folder logs/$folder_name \
      --filepaths merged_merged_beamng_udacity merged_mapelites_migration_donkey_udacity_search_mapelites_migration_donkey_beamng_search \
      --load-probability-map \
      --failure-probability \
      --multiply-probabilities

