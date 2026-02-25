

MODEL_NAME="swe-master-model-path"
vllm serve ${MODEL_NAME} \
    --tensor-parallel-size=4 \
    --gpu-memory-utilization 0.9 \
    --served-model-name swe-master-sft \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --enable-prefix-caching \
