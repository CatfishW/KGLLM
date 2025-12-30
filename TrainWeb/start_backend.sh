#!/bin/bash
cd /data/Yanlai/KGLLM/TrainWeb/backend
/home/benwulab/anaconda3/envs/KGLLM/bin/python -m uvicorn main:app --host 0.0.0.0 --port 32026
