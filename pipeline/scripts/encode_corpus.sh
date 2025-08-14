export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running pipeline script..."

CUDA_VISIBLE_DEVICES=0

python3 -m ViDRILL.pipeline.corpus_encoder \
    --corpus_path "/home/fit02/dien-workspace/vlsp/data_drill/maxlen450/corpus_450.csv"\
    --qdrant_path "/home/fit02/dien-workspace/vlsp/output/db_bge_450_from_bge_2048" \
    --model_name "/home/fit02/dien-workspace/vlsp/output/bge-m3-neg-top60-form-finetune" \
    --batch_size 1040