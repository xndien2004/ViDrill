# CUDA_VISIBLE_DEVICES=0,1

# BAAI/bge-reranker-v2-m3

# torchrun --nproc_per_node 2 \
# 	-m FlagEmbedding.finetune.reranker.encoder_only.base \
# 	--model_name_or_path BAAI/bge-reranker-v2-m3 \
#     --cache_dir ./cache/model \
#     --train_data /home/fit02/dien-workspace/vlsp/data_drill/train-neg-top60-finetune-rerank.json \
#     --cache_path ./cache/data \
#     --train_group_size 10 \
#     --query_max_len 128 \
#     --passage_max_len 2048 \
#     --pad_to_multiple_of 8 \
#     --knowledge_distillation False \
# 	--output_dir /home/fit02/dien-workspace/vlsp/output/bge-reranker-neg-top60-finetune1 \
#     --overwrite_output_dir \
#     --learning_rate 6e-5 \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --dataloader_drop_last True \
#     --warmup_ratio 0.1 \
#     --gradient_checkpointing \
#     --weight_decay 0.01 \
#     --deepspeed /home/fit02/dien-workspace/vlsp/ViDRILL/config/ds_zero2.json \
#     --logging_steps 1 \
#     --save_steps 500 \
#     2>&1 | tee bge-reranker-neg-top60-finetune1.log

# BAAI/bge-reranker-v2-m3

torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.reranker.encoder_only.base \
	--model_name_or_path /home/fit02/dien-workspace/vlsp/output/bge-reranker-neg-top60-finetune \
    --cache_dir ./cache/model \
    --train_data /home/fit02/dien-workspace/vlsp/data_drill/maxlen450/train_rerank_450_e5_gte.json \
    --cache_path ./cache/data \
    --train_group_size 10 \
    --query_max_len 128 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
	--output_dir /home/fit02/dien-workspace/vlsp/output/bge-reranker-450_e5_gte \
    --overwrite_output_dir \
    --learning_rate 6e-5 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed /home/fit02/dien-workspace/vlsp/ViDRILL/config/ds_zero2.json \
    --logging_steps 1 \
    --save_steps 500 \
    2>&1 | tee bge-reranker-450_e5_gte.log