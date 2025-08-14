export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running training script..."

CUDA_VISIBLE_DEVICES=0,1
# 

python3 -m ViDRILL.training.retrieval.train_gte \
	--train_data /home/fit02/dien-workspace/vlsp/data_drill/maxlen500/train-neg-top60-finetune-500.json \
    --model_name_or_path truro7/vn-law-embedding \
    --output_dir /home/fit02/dien-workspace/vlsp/output/vn-law-neg-top60-from-finetune-500 \
    --epochs 30 \
    --batch_size 24 \
    --max_seq_length 512 \
    2>&1 | tee vn-law-neg-top60-from-finetune-500.log