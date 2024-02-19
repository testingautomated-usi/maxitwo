#!/bin/bash

folder_name=$1

if test $# -lt 1 ; then echo 'Provide a folder_name in logs' ; exit 1 ; fi

if [ ! -d "logs/$folder_name" ]; then
  echo "logs/$folder_name does not exist."
  exit 1
fi

## BeamNG - Udacity
python merge_mapelites.py \
  --folder logs/$folder_name \
  --filepaths merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search \
  --load-probability-map \
  --failure-probability \
  --output-dir merged_merged_beamng_udacity

python merge_mapelites.py \
  --folder logs/$folder_name \
  --filepaths merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search \
  --load-probability-map \
  --failure-probability \
  --multiply-probabilities \
  --output-dir merged_merged_beamng_udacity

python merge_mapelites.py \
  --folder logs/$folder_name \
  --filepaths merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search \
  --load-probability-map \
  --failure-probability \
  --weighted-average-probabilities \
  --output-dir merged_merged_beamng_udacity


