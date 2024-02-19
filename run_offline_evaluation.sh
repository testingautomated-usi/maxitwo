sim_or_real=$1
model_architecture=$2

if [[ "$sim_or_real" == "sim" ]]; then

  echo "Beamng vs Donkey"
  python compute_offline_metrics.py --archive-path logs \
        --archive-names offline-evaluation-beamng-"$model_architecture".npz offline-evaluation-donkey-"$model_architecture".npz \
        --model-name "$model_architecture"

  echo "Udacity vs Donkey"
  python compute_offline_metrics.py --archive-path logs \
        --archive-names offline-evaluation-udacity-"$model_architecture".npz offline-evaluation-donkey-"$model_architecture".npz \
        --model-name "$model_architecture"

  echo "Beamng and Udacity vs Donkey"
  python compute_offline_metrics.py --archive-path logs \
        --archive-names offline-evaluation-beamng-"$model_architecture".npz offline-evaluation-udacity-"$model_architecture".npz offline-evaluation-donkey-"$model_architecture".npz \
        --model-name "$model_architecture"

elif [[ "$sim_or_real" == "real" ]]; then

  echo "Beamng vs Donkey"
  python compute_offline_metrics.py --archive-path logs \
        --archive-names offline-evaluation-fake-beamng-"$model_architecture".npz offline-evaluation-fake-donkey-"$model_architecture".npz \
        --model-name "$model_architecture"

  echo "Udacity vs Donkey"
  python compute_offline_metrics.py --archive-path logs \
        --archive-names offline-evaluation-fake-udacity-"$model_architecture".npz offline-evaluation-fake-donkey-"$model_architecture".npz \
        --model-name "$model_architecture"

  echo "Beamng and Udacity vs Donkey"
  python compute_offline_metrics.py --archive-path logs \
        --archive-names offline-evaluation-fake-beamng-"$model_architecture".npz offline-evaluation-fake-udacity-"$model_architecture".npz offline-evaluation-fake-donkey-"$model_architecture".npz \
        --model-name "$model_architecture"

else

  echo Unknown sim or real: "$sim_or_real"
  exit 1

fi







