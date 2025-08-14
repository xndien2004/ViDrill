export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running training script..."

export CUDA_VISIBLE_DEVICES=0,1


torchrun --nproc_per_node=2 -m ViDRILL.training.llm.train_grpo \
    --dataset_path "/home/fit02/dien-workspace/vlsp/data_drill/maxlen450/train_rerank_450_e5_gte.json" \
    --topk_neg 10 \
    --output_dir "/home/fit02/dien-workspace/vlsp/output/Qwen-1.5B-Instruct-GRPO" \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --deepspeed_path "/home/fit02/dien-workspace/vlsp/ViDRILL/config/ds_zero2.json" \
    --learning_rate 1e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --lr_scheduler_type linear \
    --logging_steps 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 2 \
    --max_prompt_length 7168 \
    --max_completion_length 512 \
    --num_train_epochs 8 \
    --save_steps 500 \
    --log_on_each_node \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    2>&1 | tee Qwen-1.5B-Instruct-GRPO.log