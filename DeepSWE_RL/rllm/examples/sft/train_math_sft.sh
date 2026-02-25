CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train_math_sft.py \
    model.partial_pretrain=Qwen/Qwen2.5-Math-1.5B \
    model.trust_remote_code=true \
    trainer.total_epochs=2 \
    data.train_batch_size=2 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=6144 \
    data.truncation=right \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_files=large_sft_data.parquet \
    data.val_files=sft_data.parquet \
    trainer.default_local_dir=outputs/qwen2.5_math_sft
