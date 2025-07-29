# Create a conda virtual environment
conda create -n dualbind python=3.10 

# Activate the env
conda activate dualbind

# Install key dependencies
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install lightning -c conda-forge
conda install -c conda-forge rdkit
pip install biopython
pip install chemprop
pip install hydra-core
pip install omegaconf
pip install biotite
