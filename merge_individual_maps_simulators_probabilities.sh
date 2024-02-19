#!/bin/bash

folder_name=$1

if test $# -lt 1 ; then echo 'Provide a folder_name in logs' ; exit 1 ; fi

if [ ! -d "logs/$folder_name" ]; then
  echo "logs/$folder_name does not exist."
  exit 1
fi

# FIXME: Cross-validation not supported yet

# BEAMNG
python merge_maps_simulator.py --env-name beamng --folder logs/$folder_name \
    --failure-probability --filepaths mapelites_beamng_search mapelites_migration_beamng_udacity_search

## UDACITY
python merge_maps_simulator.py --env-name udacity --folder logs/$folder_name \
    --failure-probability --filepaths mapelites_udacity_search mapelites_migration_udacity_beamng_search

## DONKEY
python merge_maps_simulator.py --env-name donkey --folder logs/$folder_name \
    --failure-probability --filepaths mapelites_migration_donkey_udacity_search mapelites_migration_donkey_beamng_search
