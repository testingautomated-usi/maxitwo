model_name=$1
env_name=$2
individuals_folder=$3


if [[ "$env_name" == "donkey" ]]; then
  python run_individual_migration.py --env-name donkey \
    --donkey-exe-path ..\\..\\Desktop\\DonkeySimWindowsRepl\\donkey_sim.exe \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --feature-combination turns_count-curvature \
    --filepath "$individuals_folder" \
    --cyclegan-experiment-name donkey \
    --cyclegan-checkpoints-dir cyclegan/checkpoints \
    --cyclegan-epoch 40 \
    --gpu-ids 0 \

elif [[ "$env_name" == "beamng" ]]; then
  python run_individual_migration.py --env-name beamng \
    --beamng-home ..\\..\\Desktop\\beamng_previous\\BeamNG.drive-0.23.5.1.12888 \
    --beamng-user ..\\..\\Desktop\\beamng_previous \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --feature-combination turns_count-curvature \
    --filepath "$individuals_folder" \
    --cyclegan-experiment-name beamng \
    --cyclegan-checkpoints-dir cyclegan/checkpoints \
    --cyclegan-epoch 35 \
    --gpu-ids -1 \


elif [[ "$env_name" == "udacity" ]]; then
  python run_individual_migration.py --env-name udacity \
    --udacity-exe-path ..\\..\\Desktop\\UdacitySimWindowsRepl\\self_driving_car_nanodegree_program.exe \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --feature-combination turns_count-curvature \
    --filepath "$individuals_folder" \
    --cyclegan-experiment-name udacity \
    --cyclegan-checkpoints-dir cyclegan/checkpoints \
    --cyclegan-epoch 35 \
    --gpu-ids 0 \

else
  echo Unknown env_name: "$env_name"
  exit 1
fi







