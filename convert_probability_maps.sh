#!/bin/bash

folder_name=$1

if test $# -lt 1 ; then echo 'Provide a folder_name in logs' ; exit 1 ; fi

if [ ! -d "logs/$folder_name" ]; then
  echo "logs/$folder_name does not exist."
  exit 1
fi

# BEAMNG
python convert_probability_map.py --folder logs/"$folder_name" --filepath mapelites_beamng_search --failure-probability
python convert_probability_map.py --folder logs/"$folder_name" --filepath mapelites_migration_udacity_beamng_search --failure-probability
python convert_probability_map.py --folder logs/"$folder_name" --filepath mapelites_migration_donkey_beamng_search --failure-probability

## UDACITY
python convert_probability_map.py --folder logs/"$folder_name" --filepath mapelites_udacity_search --failure-probability
python convert_probability_map.py --folder logs/"$folder_name" --filepath mapelites_migration_beamng_udacity_search --failure-probability
python convert_probability_map.py --folder logs/"$folder_name" --filepath mapelites_migration_donkey_udacity_search --failure-probability

