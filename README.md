| :zap:        Complete replication package coming soon!        :zap: |
|---------------------------------------------------------------------|

# Replication Package for the paper: "Two is Better Than One: Digital Siblings to Improve Autonomous Driving Testing"

The repository contains the source code to run the experiments and analyze the results.
The README only provides instruction on how to replicate the experiments by using pretrained models.
The source code of the simulators (i.e., **DonkeyCar** and **Udacity**) as well as the datasets to train the models 
(both cyclegan and dave2) will be provided on request. 

As for the **BeamNG** simulator we used the version _0.23.5.1.1288_. 
A free version of the BeamNG simulator for research purposes can be obtained by registering at [beamng.tech](https://register.beamng.tech) and following the instructions provided by BeamNG.
If the above version is not available you can have a look at the code from the [SBST CPS Tool competition](https://github.com/sbft-cps-tool-competition/cps-tool-competition)
which uses a recent version of BeamNG. The code in the package [code_pipeline](https://github.com/testingautomated-usi/maxitwo/tree/master/code_pipeline) and [envs/beamng](https://github.com/testingautomated-usi/maxitwo/tree/master/envs/beamng) have to be updated 
to reflect the changes of the **beamngpy** library that communicates with the simulator. Such library has to be updated in the [requirements.txt](https://github.com/testingautomated-usi/maxitwo/tree/master/requirements.txt) file.

The link to download the pretrained models as well as the results of the experiment can be downloaded [here](https://drive.switch.ch/index.php/s/fMkAVQSCO5plOBZ).
This needs to be done to execute the commands after step 0.

## Step 0. Configure the environment:

1) Install [anaconda](https://docs.anaconda.com/anaconda/install/) for your system;

2) Create the environment: `conda create -n myenv python=3.8`

3) Activate the environment: `conda activate myenv`

4) Install the requirements: `pip install -r requirements.txt`

## Analyzing the results

Copy the `logs` folder in the archive downloaded before and place it at the root of the folder containing this repository.

### 1. Failure probabilities

Type:

```commandline
./compute_correlations_probabilities.sh mapelites
```

to compute the correlations using the model trained with simulation data and type:

```commandline
./compute_correlations_probabilities.sh cyclegan
```

to compute the correlations using the model trained using the cyclegan translation.

### 2. Quality metrics

Type:

```commandline
./compute_correlations_quality_metrics.sh mapelites
```

to compute the correlations using the model trained with simulation data and type:

```commandline
./compute_correlations_quality_metrics.sh cyclegan
```

to compute the correlations using the model trained using the cyclegan translation.

## Replicating the experiments

### 1. Run the MapElites algorithm

#### 1.1. Using the dave2 model trained with simulated images

The command to type for `Udacity` is:

```commandline
python run_mapelites.py --env-name udacity \
    --udacity-exe-path <path/to/udacity-simulator/executable> \
    --agent-type supervised \
    --model-path <path/to/mixed-dave2.h5> \
    --min-angle 100 \
    --max-angle 300 \
    --population-size 20 \
    --iteration-runtime 150 \
    --mutation-extent 6 \
    --feature-combination turns_count-curvature \
    --num-runs 5
```

The simulator executable as well as the dave2 model (i.e., `mixed-dave2.h5`) 
are provided in the previously downloaded archive (respectively under `simulators` and `models`).

The command above runs the MapElites algorithm in the `Udacity` simulator using the
dave2 model trained with simulated data for 5 runs, each of 150 iterations. 
The command creates a folder for each run in the `logs` folder named `mapelites_udacity_<date_str>_i` where `i` is the index of the run. 
Moreover it creates a folder with all the individuals executed in each run, namely `mapelites_udacity_<date_str>_all`.

The MapElites algorithm needs to be executed for the same number of runs and iterations using the `BeamNG` simulator. The command is the following:

```commandline
python run_mapelites.py --env-name beamng \
    --beamng-home <path/to/beamng-simulator/BeamNG.drive-0.23.5.1.12888> \
    --beamng-user <path/to/beamng-simulator> \
    --agent-type supervised \
    --model-path <path/to/mixed-dave2.h5> \
    --min-angle 100 \
    --max-angle 300 \
    --population-size 20 \
    --iteration-runtime 150 \
    --mutation-extent 6 \
    --feature-combination turns_count-curvature \
    --num-runs 5
```

#### 1.2. Using the dave2 model trained with pseudo-real images

In this case we need to use the pretrained cyclegan model to translate the 
images coming from the simulator to pseudo-real images. 
The command is similar to the one above except fo the cyclegan related parameters:

```commandline
python run_mapelites.py --env-name udacity \
    --udacity-exe-path <path/to/udacity-simulator/executable> \
    --agent-type supervised \
    --model-path <path/to/mixed-fake-dave2.h5> \
    --min-angle 100 \
    --max-angle 300 \
    --population-size 20 \
    --iteration-runtime 150 \
    --mutation-extent 6 \
    --feature-combination turns_count-curvature \
    --num-runs 5 \
    --cyclegan-experiment-name udacity \
    --gpu-ids 0 \
    --cyclegan-checkpoints-dir cyclegan/checkpoints \
    --cyclegan-epoch 35
```

This is the command for `Udacity`. Similarly for `BeamNG` the 
`--cyclegan-experiment-name` parameter is `beamng` and the 
`--cyclegan-epoch` parameter is 35. For `Donkey` the 
`--cyclegan-experiment-name` parameter is `donkey` and the 
`--cyclegan-epoch` parameter is 40. In the commands below we only provide examples for the model trained using simulated images; 
cyclegan related parameters need to be provided when the model under test is `mixed-fake-dave2.h5` as shown above.

The command above needs to be executed on a machine with GPU as the inference time needs to match the 
frame rate of the simulator. The `checkpoints` folder of the cyclegan should have already been downloaded; 
the archive contains a folder named `cyclegan` with a sub-folder named `checkpoints`. Such folder needs to be placed
in the `cyclegan` folder at the root of the folder containing this repository.

### 2. Migration

#### 2.1. Migrating individuals found by running MapElites on BeamNG on Udacity (Simulated images)

```commandline
python run_individual_migration.py --env-name udacity \
    --udacity-exe-path <path/to/udacity-simulator/executable> \
    --agent-type supervised \
    --model-path <path/to/mixed-dave2.h5> \
    --feature-combination turns_count-curvature \
    --filepath mapelites_beamng_<date_str>_all
```

This command will create the folder `mapelites_migration_udacity_<date_str>`.
It is better to rename such folder in `mapelites_migration_udacity_beamng_search`. 
It should be read in this way: migration on `Udacity` of individuals obtained by running the search (i.e., MapElites) on `BeamNG`. The command is for simulated images; for pseudo-real images the three options showed in Section 1.2. should be added.

#### 2.2. Migrating individuals found by running MapElites on Udacity on BeamNG (Simulated images)

```commandline
python run_individual_migration.py --env-name beamng \
    --beamng-home <path/to/beamng-simulator/BeamNG.drive-0.23.5.1.12888> \
    --beamng-user <path/to/beamng-simulator> \
    --agent-type supervised \
    --model-path <path/to/mixed-dave2.h5> \
    --feature-combination turns_count-curvature \
    --filepath mapelites_udacity_<date_str>_all
```

This command will create the folder `mapelites_migration_beamng_<date_str>`.
It is better to rename such folder in `mapelites_migration_beamng_udacity_search`. 
It should be read in this way: migration on `BeamNG` of individuals obtained by running the search (i.e., MapElites) on `Udacity`. The command is for simulated images; for pseudo-real images the three options showed in Section 1.2. should be added.

#### 2.3. Migrating individuals found by running MapElites on BeamNG on Donkey (Simulated images)

```commandline
python run_individual_migration.py --env-name donkey \
    --donkey-exe-path <path/to/donkey-simulator/executable> \
    --agent-type supervised \
    --model-path <path/to/mixed-dave2.h5> \
    --feature-combination turns_count-curvature \
    --filepath mapelites_beamng_<date_str>_all
```

This command will create the folder `mapelites_migration_donkey_<date_str>`.
It is better to rename such folder in `mapelites_migration_donkey_beamng_search`. 
It should be read in this way: migration on `Donkey` of individuals obtained by running the search (i.e., MapElites) on `BeamNG`. The command is for simulated images; for pseudo-real images the three options showed in Section 1.2. should be added.

#### 2.4. Migrating individuals found by running MapElites on Udacity on Donkey (Simulated images)

```commandline
python run_individual_migration.py --env-name donkey \
    --donkey-exe-path <path/to/donkey-simulator/executable> \
    --agent-type supervised \
    --model-path <path/to/mixed-dave2.h5> \
    --feature-combination turns_count-curvature \
    --filepath mapelites_udacity_<date_str>_all
```

This command will create the folder `mapelites_migration_donkey_<date_str>`.
It is better to rename such folder in `mapelites_migration_donkey_udacity_search`. 
It should be read in this way: migration on `Donkey` of individuals obtained by running the search (i.e., MapElites) on `Udacity`. The command is for simulated images; for pseudo-real images the three options showed in Section 1.2. should be added.

### 3. Union

In this step we merge the maps coming from the same simulators. This step is called Union in the paper.

First, copy the `mapelites_run/mapelites_beamng_<date_str>_all` folder and rename it as `mapelites_beamng_search`;
in the same way copy the `mapelites_run/mapelites_udacity_<date_str>_all` folder and rename it as `mapelites_udacity_search`.

Then, create a folder under `logs` grouping the `_all` folders of the MapElites search and migrations on `Udacity`, `BeamNG` and `Donkey`.
For the sake of the example let us call the folder `mapelites_run`.

At this point the `mapelites_run` folder under `logs` should contain the following folders:

```commandline
mapelites_beamng_<date_str>_0
mapelites_beamng_<date_str>_1
mapelites_beamng_<date_str>_2
mapelites_beamng_<date_str>_3
mapelites_beamng_<date_str>_4
mapelites_beamng_<date_str>_all
mapelites_beamng_search
mapelites_migration_beamng_udacity_search
mapelites_migration_donkey_beamng_search
mapelites_migration_donkey_udacity_search
mapelites_migration_udacity_beamng_search
mapelites_udacity_<date_str>_0
mapelites_udacity_<date_str>_1
mapelites_udacity_<date_str>_2
mapelites_udacity_<date_str>_3
mapelites_udacity_<date_str>_4
mapelites_udacity_<date_str>_all
mapelites_udacity_search
```

while the `logs` folder contains:

```commandline
mapelites_run
```

### 3.1. Convert Maps

Run the following command:

```commandline
./convert_probability_maps.sh mapelites_run
```

to convert the maps from success probability (generated by the search/migration) to failure probability required by the following analyses. 

### 3.2. Probability Maps

Run the following command:

```commandline
./merge_individual_maps_simulators_probabilities.sh mapelites_run
```

The command creates the following folders in `mapelites_run`:

```commandline
merged_mapelites_beamng_search_mapelites_migration_beamng_udacity_search
merged_mapelites_migration_donkey_udacity_search_mapelites_migration_donkey_beamng_search
merged_mapelites_udacity_search_mapelites_migration_udacity_beamng_search
```

At this point all maps contain the same individuals but executed on different simulators, i.e., 
respectively `BeamNG`, `Donkey` and `Udacity`.

### 3.3. Quality Metrics Maps

Run the following command:

```commandline
python plot_quality_metrics_map.py --folder logs \
    --filepath mapelites_run \
    --datetime-str-beamng <date_str_beamng> \
    --datetime-str-udacity <date_str_udacity> 
```

where `<date_str_beamng>` and `<date_str_udacity>` are the datetime strings 
corresponding to the folders created by the MapElites search respectively for `BeamNG` and `Udacity`.

Move the quality metrics maps from `mapelites_udacity_<date_str>_all` to `mapelites_run/mapelites_udacity_search` as well as
move the quality metrics maps from `mapelites_beamng_<date_str>_all` to `mapelites_run/mapelites_beamng_search`.

The maps are the following:

```commandline
raw_heatmap_max_lateral_positions_turns_count_curvature_iterations_0.json
raw_heatmap_std_lateral_positions_turns_count_curvature_iterations_0.json
raw_heatmap_std_speeds_turns_count_curvature_iterations_0.json
raw_heatmap_std_steering_angles_turns_count_curvature_iterations_0.json
```

The `.png` files can be copied as well but it is not mandatory.

The previous command also prints on the console the bounds for each quality metric. For instance:

```commandline
INFO:plot_quality_metrics_map:Bounds for the quality metrics: 
{
    'std_steering_angles': (0.060142596293090676, 0.5921246225056953), 
    'std_lateral_positions': (0.05692786404029756, 0.8862904614683409), 
    'std_speeds': (4.6374119298227985, 12.133422056281063), 
    'max_lateral_positions': (0.20460000000000012, 2.1)
}
```

The first number in each tuple is the lower bound for the respective metric and the second is the upper bound.

Then run the following command:

```commandline
./merge_individual_maps_simulators_quality_metrics.sh mapelites_run \
    0.060142596293090676 0.5921246225056953 \
    0.05692786404029756 0.8862904614683409 \
    4.6374119298227985 12.133422056281063 \
    0.20460000000000012 2.1
```

The bounds come from the previous step and need to be provided in the order specified in the previous step.

### 4. Merge

In this step we merge the maps coming from different simulators.

### 4.1. Probability Maps

Run the following command:

```commandline
./merge_maps_simulators_probabilities.sh mapelites_run
```

The command merges the failure probability maps of `BeamNG` and `Udacity` using the product operator.
The command should produce the folder `merged_merged_beamng_udacity` in `mapelites_run`.

### 4.2. Quality Metrics Maps

Run the following command:

```commandline
./merge_maps_simulators_quality_metrics.sh mapelites_run min \
    0.060142596293090676 0.5921246225056953 \
    0.05692786404029756 0.8862904614683409 \
    4.6374119298227985 12.133422056281063 \
    0.20460000000000012 2.1
```

The command merges the quality metrics maps of `BeamNG` and `Udacity` using the minimum operator.
The command should produce the quality metrics maps inside `mapelites_run/merged_merged_beamng_udacity`.

At the end of such step the content of the folder `mapelites_run/merged_merged_beamng_udacity` should be the following:

```
heatmap_failure_probability_multiply_turns_count_curvature_iterations_0.png
heatmap_max_lateral_positions_min_turns_count_curvature_iterations_0.png
heatmap_std_lateral_positions_min_turns_count_curvature_iterations_0.png
heatmap_std_speeds_min_turns_count_curvature_iterations_0.png
heatmap_std_steering_angles_min_turns_count_curvature_iterations_0.png
raw_heatmap_failure_probability_multiply_turns_count_curvature_iterations_0.json
raw_heatmap_max_lateral_positions_min_turns_count_curvature_iterations_0.json
raw_heatmap_std_lateral_positions_min_turns_count_curvature_iterations_0.json
raw_heatmap_std_speeds_min_turns_count_curvature_iterations_0.json
raw_heatmap_std_steering_angles_min_turns_count_curvature_iterations_0.json
```

### 5. Analysis

In this step we compute the correlations between the digital siblings (i.e., `Udacity` and `BeamNG`) and the digital twin (i.e., `Donkey`).

### 5.1. Probabilities

Run the following command:

```commandline
./compute_correlations_probabilities.sh mapelites_run
```

The correlations and all the other metrics are printed on console.

### 5.2. Quality Metrics

```commandline
./compute_correlations_quality_metrics.sh mapelites_run
```

The correlations and all the other metrics are printed on console. 
The command only computes the correlations for the `max_lateral_position` quality metric.

### 6. Offline Evaluation

Download the offline evaluation datasets from [here](https://drive.switch.ch/index.php/s/4GyponKp6hx1Rs6) and copy them within the `logs` folder of this repository. Let us carry out the offline evaluation for the `dave2` model.

### 5.1. Simulated images

From the root of the project type:

```commandline
python compute_offline_metrics.py --archive-path logs \
	--archive-names offline-evaluation-beamng-dave2 offline-evaluation-donkey-dave2.npz \
	--show-plot
```

to compute the correlation and the distance between the prediction error distributions of `BeamNG` and `Donkey`. Save the plot where desired.

Type:

```commandline
python compute_offline_metrics.py --archive-path logs \
	--archive-names offline-evaluation-udacity-dave2 offline-evaluation-donkey-dave2.npz \
	--show-plot
```

to compute the correlation and the distance between the prediction error distributions of `Udacity` and `Donkey`. Save the plot where desired.

Type:

```commandline
python compute_offline_metrics.py --archive-path logs \
	--archive-names offline-evaluation-beamng-dave2 offline-evaluation-udacity-dave2 offline-evaluation-donkey-dave2.npz \
	--show-plot
```

to merge the prediction error distributions of `BeamNG` and `Udacity` and compute its correlation and distance w.r.t. the prediction error distribution of `Donkey`. Save the plot where desired.

### 5.2. Pseudo-real images

From the root of the project type:

```commandline
python compute_offline_metrics.py --archive-path logs \
	--archive-names offline-evaluation-fake-beamng-dave2 offline-evaluation-fake-donkey-dave2.npz \
	--show-plot
```

to compute the correlation and the distance between the prediction error distributions of `BeamNG` and `Donkey`. Save the plot where desired.

Type:

```commandline
python compute_offline_metrics.py --archive-path logs \
	--archive-names offline-evaluation-fake-udacity-dave2 offline-evaluation-fake-donkey-dave2.npz \
	--show-plot
```

to compute the correlation and the distance between the prediction error distributions of `Udacity` and `Donkey`. Save the plot where desired.

Type:

```commandline
python compute_offline_metrics.py --archive-path logs \
	--archive-names offline-evaluation-fake-beamng-dave2 offline-evaluation-fake-udacity-dave2 offline-evaluation-fake-donkey-dave2.npz \
	--show-plot
```

to merge the prediction error distributions of `BeamNG` and `Udacity` and compute its correlation and distance w.r.t. the prediction error distribution of `Donkey`. Save the plot where desired.






