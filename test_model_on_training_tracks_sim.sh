model_name=$1

python collect_images.py --env-name udacity \
    --udacity-exe-path ..\\..\\Desktop\\UdacitySimWindowsRepl\\self_driving_car_nanodegree_program.exe \
    --seed 0 --num-episodes 50 \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --max-angle 270 \
    --no-save-archive

python collect_images.py --env-name donkey \
    --donkey-exe-path ..\\..\\Desktop\\DonkeySimWindowsRepl\\donkey_sim.exe \
    --seed 0 --num-episodes 50 \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --max-angle 270 \
    --no-save-archive

python collect_images.py --env-name beamng \
    --beamng-home ..\\..\\Desktop\\beamng_previous\\BeamNG.drive-0.23.5.1.12888 \
    --beamng-user ..\\..\\Desktop\\beamng_previous --seed 0 --num-episodes 50 \
    --agent-type supervised \
    --model-path logs\\models\\"$model_name" \
    --max-angle 270 \
    --no-save-archive







