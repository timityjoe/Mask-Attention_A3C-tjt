#!/bin/bash

# If encountering "Your shell has not been properly configured to use 'conda activate'.", then "conda deactivate" from (base)
# source activate base	
# conda deactivate
	
# conda init bash
# conda create --name conda36-maskattn python=3.6.8
# conda activate conda39-maskattn
# pip install -r requirements.txt
# conda deactivate
# conda clean --all	# Purge cache and unused apps
# condo info


# See https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html
echo "Setting up dectransformer Environment..."
source activate base	
conda deactivate
conda activate conda36-maskattn
echo "$PYTHON_PATH"
