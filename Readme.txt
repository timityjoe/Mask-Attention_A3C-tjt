
# If encountering "Your shell has not been properly configured to use 'conda activate'.", then "conda deactivate" from (base)
source activate base	
conda deactivate
	
conda init bash
conda create --name conda36-maskattn python=3.6.8
conda activate conda39-maskattn
pip install -r requirements.txt
conda deactivate
conda clean --all	# Purge cache and unused apps
condo info

# Tensorflow with Cuda GPU install (Conda)
# See
# https://utho.com/docs/tutorial/how-to-install-anaconda-on-ubuntu-20-04-lts/
# https://www.tensorflow.org/install/pip
conda install cuda -c nvidia
conda install -c nvidia cudnn
	* Set these env variables after CUDNN is installed
	CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
	
# Install Atari ROM:
# https://github.com/openai/atari-py#roms
python -m atari_py.import_roms <path to folder>

# Torch not compiled with CUDA enabled ubuntu
# https://www.datasciencelearner.com/assertionerror-torch-not-compiled-with-cuda-enabled-fix/
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --upgrade pip
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113^C

source maskattn.sh
conda activate conda36-maskattn


# From
# https://github.com/machine-perception-robotics-group/Mask-Attention_A3C/blob/main/README.md
python3 experiment.py --env hopper --dataset medium --model_type dt -w

python3 gym_eval.py --convlstm --mask_double --env BreakoutNoFrameskip-v4 --load-model BreakoutNoFrameskip-v4_Mask-A3C-double+ConvLSTM_best --num-episodes 100

python3 gym_eval.py --convlstm --mask_double --env BreakoutNoFrameskip-v4 --load-model BreakoutNoFrameskip-v4_Mask-A3C-double+ConvLSTM_best --num-episodes 100 --gpu-ids 0 --image

python3 gym_eval.py --convlstm --mask_double --env BreakoutNoFrameskip-v4 --load-model BreakoutNoFrameskip-v4_Mask-A3C-double+ConvLSTM_best --num-episodes 100 --gpu-ids 0 --render


sh ./start_eval.sh

# Tensorboard 
tensorboard --logdir=./ --port=8080









