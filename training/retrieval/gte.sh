export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running training script..."

CUDA_VISIBLE_DEVICES=0,1
# 

python3 -m ViDRILL.training.retrieval.train_gte \
	--train_data /home/fit02/dien-workspace/vlsp/data_drill/maxlen450/train_rerank_450.csv \
    --model_name_or_path Alibaba-NLP/gte-multilingual-base \
    --output_dir /home/fit02/dien-workspace/vlsp/output/gte-450 \
    --epochs 30 \
    --batch_size 24 \
    --max_seq_length 512 \
    2>&1 | tee gte-450.log