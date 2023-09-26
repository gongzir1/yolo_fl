#!/bin/bash

# Activate the Conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate yolo

cd /home/gongzir/project/fedlearn
# Run the Python files
python runner_server.py --server_address [::]:8080 --device cuda:2 --rounds 2 --load_params
#python runner_client.py --server_address [::]:8080 --device cuda:0 --cid 1 --epochs 10
#python runner_client.py --server_address [::]:8080 --device cuda:1 --cid 2 --epochs 10

# Deactivate the Conda environment
#conda deactivate

