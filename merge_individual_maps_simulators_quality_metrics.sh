#!/usr/local/bin/bash
# installed with brew install bash; here I need associative arrays that are defined in bash starting from v4. Version of bash on the mac (i.e. /bin/bash) is v3
# https://stackoverflow.com/questions/6047648/associative-arrays-error-declare-a-invalid-option

folder_name=$1
min_bound_sa=$2
max_bound_sa=$3
min_bound_lp=$4
max_bound_lp=$5
min_bound_sp=$6
max_bound_sp=$7
min_bound_max_lp=$8
max_bound_max_lp=$9

if test $# -lt 9 ; then echo 'Provide a folder_name in logs | Min bound std_steering_angles | Max bound std_steering_angles Min bound std_lateral_positions | Max bound std_lateral_positions | Min bound std_speeds | Max bound std_speeds | Min bound max_lateral_positions | Max bound max_lateral_positions' ; exit 1 ; fi

if [ ! -d "logs/$folder_name" ]; then
  echo "logs/$folder_name does not exist."
  exit 1
fi

declare -A quality_metrics_bounds=(["std_steering_angles_min"]="$min_bound_sa" ["std_steering_angles_max"]="$max_bound_sa" ["std_lateral_positions_min"]="$min_bound_lp" ["std_lateral_positions_max"]="$max_bound_lp" ["std_speeds_min"]="$min_bound_sp" ["std_speeds_max"]="$max_bound_sp" ["max_lateral_positions_min"]="$min_bound_max_lp" ["max_lateral_positions_max"]="$max_bound_max_lp")

# BEAMNG
# Should synchronize with test_generators/mapelites/config.py
for quality_metric in std_steering_angles std_lateral_positions std_speeds max_lateral_positions ; do
  python merge_maps_simulator.py --env-name beamng --folder logs/$folder_name \
    --filepaths mapelites_beamng_search mapelites_migration_beamng_udacity_search \
    --quality-metric $quality_metric --min-quality-metric "${quality_metrics_bounds["$quality_metric"_min]}" --max-quality-metric "${quality_metrics_bounds["$quality_metric"_max]}"
done

# DONKEY
for quality_metric in std_steering_angles std_lateral_positions std_speeds max_lateral_positions ; do
  python merge_maps_simulator.py --env-name donkey --folder logs/$folder_name \
    --filepaths mapelites_migration_donkey_udacity_search mapelites_migration_donkey_beamng_search \
    --quality-metric $quality_metric --min-quality-metric "${quality_metrics_bounds["$quality_metric"_min]}" --max-quality-metric "${quality_metrics_bounds["$quality_metric"_max]}"
done

# UDACITY
for quality_metric in std_steering_angles std_lateral_positions std_speeds max_lateral_positions ; do
  python merge_maps_simulator.py --env-name udacity --folder logs/$folder_name \
    --filepaths mapelites_udacity_search mapelites_migration_udacity_beamng_search \
    --quality-metric $quality_metric --min-quality-metric "${quality_metrics_bounds["$quality_metric"_min]}" --max-quality-metric "${quality_metrics_bounds["$quality_metric"_max]}"
done
