export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

HEAD_NODE_IP=xxx.xxx.xxx.xxx 
HEAD_NODE_PORT=8266


ray start --address="${HEAD_NODE_IP}:${HEAD_NODE_PORT}"  --num-gpus=8 \