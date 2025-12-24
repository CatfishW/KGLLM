#!/bin/bash
# Restart vLLM Server script
# Derived from process inspection of PID 1761728

echo "Searching for existing vLLM processes..."
pids=$(ps -ef | grep "vllm serve" | grep -v grep | awk '{print $2}')

if [ -n "$pids" ]; then
    echo "Killing existing vLLM PIDs: $pids"
    kill $pids
    
    echo "Waiting for process termination..."
    sleep 10
    
    # Check if still alive
    remaining=$(ps -ef | grep "vllm serve" | grep -v grep)
    if [ -n "$remaining" ]; then
        echo "Process stuck. Force killing..."
        kill -9 $pids
        sleep 5
    fi
else
    echo "No running vLLM found."
fi

echo "Starting vLLM Server..."
# Use absolute paths to ensure correct environment
PYTHON_BIN="/home/benwulab/anaconda3/envs/vllm/bin/python3.12"
VLLM_BIN="/home/benwulab/anaconda3/envs/vllm/bin/vllm"

# Ensure CUDA devices are visible (Using all GPUs as per original config TP=2)
export CUDA_VISIBLE_DEVICES=0,1

nohup $PYTHON_BIN $VLLM_BIN serve Qwen/Qwen3-VL-32B-Instruct-FP8 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.63 \
    --max-model-len 38000 \
    --port 8888 \
    --async-scheduling \
    --max_num_seqs 4 \
    --enable-chunked-prefill \
    > vllm_restart_manual.log 2>&1 &

echo "vLLM Server launched! PID: $!"
echo "Log: vllm_restart_manual.log"
