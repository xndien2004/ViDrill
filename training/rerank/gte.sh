export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running training script..."

export CUDA_VISIBLE_DEVICES=0,1
# 

torchrun --nproc_per_node=2 \
    -m ViDRILL.training.rerank.train_gte \
	--train_file /home/fit02/dien-workspace/vlsp/data_drill/train-neg-top60-finetune.json \
    --model_name Alibaba-NLP/gte-multilingual-reranker-base \
    --output_dir /home/fit02/dien-workspace/vlsp/output/gte-rerank-neg-top60-from-finetune \
    --num_epoch 50 \
    --batch_size 20 \
    --lr 2e-5 \
    2>&1 | tee gte-rerank-neg-top60-from-finetune.log