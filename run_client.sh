#!/bin/bash
#DATASET=$1
NUM_CLIENT=$1
#NUM_CLIENT=10
START_PORT=1111
#MODEL=$3
#PORT=$4
# Activate the Conda environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate yolo

cd /home/gongzir/project/fedlearn
# Run the Python files
#python runner_client.py --server_address [::]:8080 --device cuda:2 --rounds 2 --load_params
#python runner_client.py --server_address [::]:8080 --device cuda:0 --cid 1 --epochs 10 --data data/meat.yaml
#python runner_client.py --server_address [::]:8080 --device cuda:1 --cid 2 --epochs 10 --data data/meat1.yaml




if [ ! -n "$NUM_CLIENT" ];then
	echo "Please input num of client"
	exit
fi

#if [ ! -n "$EPOCHS" ];then
#	echo "Please input epochs"
#	exit
#fi


#if [ ! -n "$PORT" ];then
#	echo "please input server port"
#	exit
#fi
#--data data/meat$((i-1)).yaml \
#--device cuda:$((($i % 8)))\
for i in $(seq 1 ${NUM_CLIENT}); do
  CID=$i
#  CUDA_DEVICE=$((i % 8))
#  PORT=$((START_PORT + i))
#	nohup python3 -m torch.distributed.run --master_port $PORT --nproc_per_node 8 runner_client.py \
  nohup python3 runner_client.py \
         --cid $CID \
         --epochs 10 \
         --data data/voc$(i).yaml \
         --server_address [::]:8080 &
done

while true; do
    echo "This is a sentence printed every 13 minutes."
    sleep 800
done

