model_name=$1
env_name=$2
resume_datestr=$3
resume_run=$4

if [[ "$env_name" == "beamng" ]]; then
  if [ -n "$resume_datestr" ] && [ -n "$resume_run" ]; then
    python run_mapelites.py --env-name beamng \
    --beamng-home ..\\..\\Desktop\\beamng_previous\\BeamNG.drive-0.23.5.1.12888 \
    --beamng-user ..\\..\\Desktop\\beamng_previous \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --min-angle 100 \
    --max-angle 200 \
    --population-size 20 \
    --mutation-extent 6 \
    --iteration-runtime 100 \
    --feature-combination turns_count-curvature \
    --num-runs 5 \
    --cyclegan-experiment-name beamng \
    --cyclegan-checkpoints-dir cyclegan/checkpoints \
    --cyclegan-epoch 35 \
    --gpu-ids -1 \
    --resume-datestr "$resume_datestr" \
    --resume-run "$resume_run"
  else
    python run_mapelites.py --env-name beamng \
    --beamng-home ..\\..\\Desktop\\beamng_previous\\BeamNG.drive-0.23.5.1.12888 \
    --beamng-user ..\\..\\Desktop\\beamng_previous \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --min-angle 100 \
    --max-angle 200 \
    --population-size 20 \
    --mutation-extent 6 \
    --iteration-runtime 100 \
    --feature-combination turns_count-curvature \
    --num-runs 5 \
    --cyclegan-experiment-name beamng \
    --cyclegan-checkpoints-dir cyclegan/checkpoints \
    --cyclegan-epoch 35 \
    --gpu-ids -1
  fi
elif [[ "$env_name" == "udacity" ]]; then
  if [ -n "$resume_datestr" ] && [ -n "$resume_run" ]; then
    python run_mapelites.py --env-name udacity \
    --udacity-exe-path ..\\..\\Desktop\\UdacitySimWindowsRepl\\self_driving_car_nanodegree_program.exe \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --min-angle 100 \
    --max-angle 200 \
    --population-size 20 \
    --mutation-extent 6 \
    --iteration-runtime 100 \
    --feature-combination turns_count-curvature \
    --num-runs 5 \
    --cyclegan-experiment-name udacity \
    --cyclegan-checkpoints-dir cyclegan/checkpoints \
    --cyclegan-epoch 35 \
    --gpu-ids 0 \
    --resume-datestr "$resume_datestr" \
    --resume-run "$resume_run"
  else
    python run_mapelites.py --env-name udacity \
    --udacity-exe-path ..\\..\\Desktop\\UdacitySimWindowsRepl\\self_driving_car_nanodegree_program.exe \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --min-angle 100 \
    --max-angle 200 \
    --population-size 20 \
    --mutation-extent 6 \
    --iteration-runtime 100 \
    --feature-combination turns_count-curvature \
    --num-runs 5 \
    --cyclegan-experiment-name udacity \
    --cyclegan-checkpoints-dir cyclegan/checkpoints \
    --cyclegan-epoch 35 \
    --gpu-ids 0
  fi
else
  echo Unknown env_name: "$env_name"
  exit 1
fi







