#!/bin/bash

folder_name=$1

if test $# -lt 1; then echo 'Provide a folder_name in logs'; exit 1 ; fi

if [ ! -d "logs/$folder_name" ]; then
  echo "logs/$folder_name does not exist."
  exit 1
fi

failure_predicts_quality_metric="no"

# Should synchronize with test_generators/mapelites/config.py
quality_metric=max_lateral_positions

echo "======= $quality_metric ======="

echo "******* Search on BeamNG + migration on Udacity and all individuals migrated on Donkey *******"
python compute_mapelites_correlation.py \
  --folder logs/$folder_name \
  --filepaths merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search merged_mapelites_migration_donkey_udacity_search_mapelites_migration_donkey_beamng_search \
  --load-probability-map \
  --failure-probability \
  --quality-metric $quality_metric \
  --failure-predicts-quality-metric "$failure_predicts_quality_metric"

echo "******* Search on Udacity + migration on BeamNG and all individuals migrated on Donkey *******"
python compute_mapelites_correlation.py \
  --folder logs/$folder_name \
  --filepaths merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search merged_mapelites_migration_donkey_udacity_search_mapelites_migration_donkey_beamng_search \
  --load-probability-map \
  --failure-probability \
  --quality-metric $quality_metric \
  --failure-predicts-quality-metric "$failure_predicts_quality_metric"

echo "******* Search on both BeamNG and Udacity (MIN) and all individuals migrated on Donkey *******"
python compute_mapelites_correlation.py \
  --folder logs/$folder_name \
  --filepaths merged_merged_beamng_udacity merged_mapelites_migration_donkey_udacity_search_mapelites_migration_donkey_beamng_search \
  --load-probability-map \
  --failure-probability \
  --quality-metric $quality_metric \
  --quality-metric-merge min \
  --failure-predicts-quality-metric "$failure_predicts_quality_metric"

echo '************************************'