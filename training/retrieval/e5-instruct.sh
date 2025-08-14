export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running training script..."

export CUDA_VISIBLE_DEVICES=0,1

python3 -m ViDRILL.training.retrieval.train_e5 \
	--train_data /home/fit02/dien-workspace/vlsp/data_drill/title/train_only_title.csv \
    --model_name_or_path intfloat/multilingual-e5-large-instruct \
    --output_dir /home/fit02/dien-workspace/vlsp/output/e5-instruct-only-title1 \
    --epochs 30 \
    --batch_size 60 \
    --max_seq_length 256 \
    2>&1 | tee e5-instruct-only-title1.log