#!/usr/bin/env bash

sudo docker run -i -t \
                --gpus all \
                --cpus 16 \
                --shm-size=60G \
                -v /home/ubuntu/EyeSeg/:/app \
                -w /app \
                utsavisionailab/cityscape:torch \
                python3 train.py \
                --framework=torch \
                --num-classes=4 \
                --output-shape=640,400,4 \
                --input-shape=640,400,1 \
                --batch-size=8 \
                --model-name=eyeseg_openeds \
                --dropout-rate=0.30 \
                --log-epoch=1 \
                --epochs=100 \
                --learn-rate=1e-4