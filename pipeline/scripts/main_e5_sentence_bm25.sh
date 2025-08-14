export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running pipeline script..."

export CUDA_VISIBLE_DEVICES=1

python3 -m ViDRILL.pipeline.main_e5_sentence_bm25 \
    --input "/home/fit02/dien-workspace/vlsp/data_drill/private_test.json" \
    --output "results.json" \
    --corpus_path "/home/fit02/dien-workspace/vlsp/data_drill/maxlen450/corpus_450.csv" \
    --bge_rerank_name "/home/fit02/dien-workspace/vlsp/output/bge-reranker-neg-top60-finetune" \
    --e5_model_name "/home/fit02/dien-workspace/vlsp/output/e5-instruct-450" \
    --gte_model_name "/home/fit02/dien-workspace/vlsp/output/gte-450" \
    --e5_qdrant_path "/home/fit02/dien-workspace/vlsp/output/db_1stage_e5_450" \
    --gte_qdrant_path "/home/fit02/dien-workspace/vlsp/output/db_1stage_gte_450" \
    --top_k_retrieval 30 \
    --threshold 0.75 \
    --top_k_bm25 200 \
    --top_else_threshold 40
