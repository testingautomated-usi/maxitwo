#!/bin/bash

# bounds for each quality metric computed with plot_quality_metrics_map.py

folder_name=$1
quality_metrics_merge=$2
min_bound_sa=$3
max_bound_sa=$4
min_bound_lp=$5
max_bound_lp=$6
min_bound_sp=$7
max_bound_sp=$8
min_bound_max_lp=$9
max_bound_max_lp=${10}

if test $# -lt 10 ; then echo 'Provide a folder_name in logs | quality_metrics_merge (avg|min|max) | Min bound std_steering_angles | Max bound std_steering_angles Min bound std_lateral_positions | Max bound std_lateral_positions | Min bound std_speeds | Max bound std_speeds | Min bound max_lateral_positions | Max bound max_lateral_positions' ; exit 1 ; fi

if [ ! -d "logs/$folder_name" ]; then
  echo "logs/$folder_name does not exist."
  exit 1
fi

# BeamNG - Udacity
python merge_mapelites.py \
  --folder logs/$folder_name \
  --filepaths merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search \
  --quality-metric std_steering_angles \
  --min-quality-metric $min_bound_sa \
  --max-quality-metric $max_bound_sa \
  --quality-metric-merge $quality_metrics_merge \
  --output-dir merged_merged_beamng_udacity

python merge_mapelites.py \
  --folder logs/$folder_name \
  --filepaths merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search \
  --quality-metric std_lateral_positions \
  --min-quality-metric $min_bound_lp \
  --max-quality-metric $max_bound_lp \
  --quality-metric-merge $quality_metrics_merge \
  --output-dir merged_merged_beamng_udacity

python merge_mapelites.py \
  --folder logs/$folder_name \
  --filepaths merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search \
  --quality-metric std_speeds \
  --min-quality-metric $min_bound_sp \
  --max-quality-metric $max_bound_sp \
  --quality-metric-merge $quality_metrics_merge \
  --output-dir merged_merged_beamng_udacity

python merge_mapelites.py \
  --folder logs/$folder_name \
  --filepaths merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search \
  --quality-metric max_lateral_positions \
  --min-quality-metric $min_bound_max_lp \
  --max-quality-metric $max_bound_max_lp \
  --quality-metric-merge $quality_metrics_merge \
  --output-dir merged_merged_beamng_udacity
