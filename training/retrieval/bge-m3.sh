CUDA_VISIBLE_DEVICES=0,1
# BAAI/bge-m3
# torchrun --nproc_per_node 2 \
# 	-m FlagEmbedding.finetune.embedder.encoder_only.base \
# 	--model_name_or_path BAAI/bge-m3 \
#     --cache_dir ./cache/model \
#     --train_data /home/fit02/dien-workspace/vlsp/data_drill/train-title-neg-top60-finetune.json \
#     --cache_path ./cache/data \
#     --train_group_size 10 \
#     --query_max_len 128 \
#     --passage_max_len 2048 \
#     --pad_to_multiple_of 8 \
#     --knowledge_distillation False \
#     --same_dataset_within_batch True \
# 	--output_dir /home/fit02/dien-workspace/vlsp/output/bge-m3-title-neg-top60-form-finetune \
#     --overwrite_output_dir \
#     --learning_rate 1e-5 \
#     --num_train_epochs 10 \
#     --per_device_train_batch_size 1 \
#     --dataloader_drop_last True \
#     --warmup_ratio 0.1 \
#     --deepspeed /home/fit02/dien-workspace/vlsp/ViDRILL/config/ds_zero2.json \
#     --logging_steps 1 \
#     --save_steps 1000 \
#     --negatives_cross_device \
#     --temperature 0.02 \
#     --sentence_pooling_method cls \
#     --normalize_embeddings True \
#     --kd_loss_type m3_kd_loss \
#     2>&1 | tee bge-m3-title-neg-top60-form-finetune.log


torchrun --nproc_per_node 2 \
	-m FlagEmbedding.finetune.embedder.encoder_only.base \
	--model_name_or_path BAAI/bge-m3 \
    --cache_dir ./cache/model \
    --train_data /home/fit02/dien-workspace/vlsp/data_drill/maxlen500/train-neg-top60-finetune-500.json \
    --cache_path ./cache/data \
    --train_group_size 20 \
    --query_max_len 128 \
    --passage_max_len 640 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --same_dataset_within_batch True \
	--output_dir /home/fit02/dien-workspace/vlsp/output/bge-m3-neg-top60-form-finetune-500 \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --deepspeed /home/fit02/dien-workspace/vlsp/ViDRILL/config/ds_zero2.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
    2>&1 | tee bge-m3-neg-top60-form-finetune-500.log