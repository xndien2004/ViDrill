export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"


MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_DATA="/home/fit02/dien-workspace/vlsp/data_drill/train-neg-top60-finetune.jsonl"
OUTPUT_DIR="/home/fit02/dien-workspace/vlsp/output/qwen2.5-1.5b-instruct-rerank"


torchrun --nproc_per_node 2 \
  -m ViDRILL.open_retrievals.pipelines.rerank \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir \
  --model_name_or_path $MODEL_NAME \
  --model_type llm \
  --causal_lm True \
  --use_lora True \
  --use_quantization_config True \
  --do_train \
  --data_name_or_path $TRAIN_DATA \
  --positive_key positive \
  --negative_key negative \
  --task_prompt "Với một câu hỏi A và một đoạn văn B, hãy xác định xem đoạn văn đó có chứa câu trả lời cho câu hỏi hay không bằng cách đưa ra dự đoán là 'Yes' hoặc 'No'." \
  --query_instruction "A: {}" \
  --document_instruction 'B: {}' \
  --learning_rate 2e-4 \
  --bf16 \
  --num_train_epochs 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 128 \
  --dataloader_drop_last True \
  --max_length 2048 \
  --query_max_length 128 \
  --save_steps 200 \
  --save_total_limit 3 \
  --train_group_size 4 \
  --logging_steps 10 \
  --temperature 0.02 \
  --deepspeed /home/fit02/dien-workspace/vlsp/ViDRILL/config/ds_zero2.json \
  2>&1 | tee qwen2.5-instruct-rerank.log