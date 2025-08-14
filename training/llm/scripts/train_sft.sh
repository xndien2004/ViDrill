export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running training script..."

export CUDA_VISIBLE_DEVICES=0,1


torchrun --nproc_per_node=2 -m ViDRILL.training.llm.train_sft \
    --dataset_path "/home/fit02/dien-workspace/vlsp/data_drill/train-neg-top60-finetune.json" \
    --topk_neg 20 \
    --output_dir "/home/fit02/dien-workspace/vlsp/output/Qwen-7B-Instruct-SFT" \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --deepspeed_path "/home/fit02/dien-workspace/vlsp/ViDRILL/config/ds_zero2.json" \
    --learning_rate 2e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --lr_scheduler_type linear \
    --logging_steps 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_input_length 40960 \
    --max_target_length 2048 \
    --num_train_epochs 20 \
    --save_steps 1000 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.1 \
    2>&1 | tee Qwen-7B-Instruct-SFT.log


torchrun --nproc_per_node=2 -m ViDRILL.training.llm.train_sft \
    --dataset_path "/home/fit02/dien-workspace/vlsp/data_drill/train-neg-top60-finetune.json" \
    --topk_neg 20 \
    --output_dir "/home/fit02/dien-workspace/vlsp/output/Qwen-14B-Instruct-SFT" \
    --model_name "Qwen/Qwen2.5-14B-Instruct" \
    --deepspeed_path "/home/fit02/dien-workspace/vlsp/ViDRILL/config/ds_zero2.json" \
    --learning_rate 2e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --lr_scheduler_type linear \
    --logging_steps 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_input_length 40960 \
    --max_target_length 2048 \
    --num_train_epochs 20 \
    --save_steps 1000 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    2>&1 | tee Qwen-14B-Instruct-SFT.log