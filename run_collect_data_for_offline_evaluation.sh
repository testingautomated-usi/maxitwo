model_name=$1
env_name=$2
sim_or_real=$3
model_architecture=$4

if [[ "$env_name" == "beamng" ]]; then
  if [[ "$sim_or_real" == "sim" ]]; then
    python offline_evaluation.py --env-name "$env_name" \
      --model-name "$model_architecture" --archive-path logs \
      --archive-name beamng-2022_07_18_16_41_18-archive-agent-autopilot-seed-1234567-episodes-20-max-angle-270-length-8.npz \
      --agent-type supervised \
      --model-path logs/models/"$model_name"
  elif [[ "$sim_or_real" == "real" ]]; then
    python offline_evaluation.py --env-name "$env_name" \
      --model-name "$model_architecture" --archive-path logs \
      --archive-name beamng-2022_07_18_16_41_18-archive-agent-autopilot-seed-1234567-episodes-20-max-angle-270-length-8-fake-35.npz \
      --agent-type supervised \
      --model-path logs/models/"$model_name"
  else
    echo Unknown sim_or_real: "$env_name"
    exit 1
  fi
elif [[ "$env_name" == "udacity" ]]; then
  if [[ "$sim_or_real" == "sim" ]]; then
    python offline_evaluation.py --env-name "$env_name" \
      --model-name "$model_architecture" --archive-path logs \
      --archive-name udacity-2022_07_18_16_50_13-archive-agent-autopilot-seed-1234567-episodes-20-max-angle-270-length-8.npz \
      --agent-type supervised \
      --model-path logs/models/"$model_name"
  elif [[ "$sim_or_real" == "real" ]]; then
    python offline_evaluation.py --env-name "$env_name" \
      --model-name "$model_architecture" --archive-path logs \
      --archive-name udacity-2022_07_18_16_50_13-archive-agent-autopilot-seed-1234567-episodes-20-max-angle-270-length-8-fake-35.npz \
      --agent-type supervised \
      --model-path logs/models/"$model_name"
  else
    echo Unknown sim_or_real: "$env_name"
    exit 1
  fi
elif [[ "$env_name" == "donkey" ]]; then
  if [[ "$sim_or_real" == "sim" ]]; then
    python offline_evaluation.py --env-name "$env_name" \
      --model-name "$model_architecture" --archive-path logs \
      --archive-name donkey-2022_07_18_17_01_45-archive-agent-autopilot-seed-1234567-episodes-20-max-angle-270-length-8.npz \
      --agent-type supervised \
      --model-path logs/models/"$model_name"
  elif [[ "$sim_or_real" == "real" ]]; then
    python offline_evaluation.py --env-name "$env_name" \
      --model-name "$model_architecture" --archive-path logs \
      --archive-name donkey-2022_07_18_17_01_45-archive-agent-autopilot-seed-1234567-episodes-20-max-angle-270-length-8-fake-40.npz \
      --agent-type supervised \
      --model-path logs/models/"$model_name"
  else
    echo Unknown sim_or_real: "$env_name"
    exit 1
  fi
else
  echo Unknown env_name: "$env_name"
  exit 1
fi







