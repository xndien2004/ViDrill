export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running training script..."

CUDA_VISIBLE_DEVICES=0

# python3 -m ViDRILL.prepare_data.build_db_corpus \
#     --corpus_path "/home/fit02/dien-workspace/vlsp/data_drill/corpus.csv"\
#     --bge_qdrant_path "/home/fit02/dien-workspace/vlsp/output/bge_qdrant" \
#     --e5_qdrant_path "/home/fit02/dien-workspace/vlsp/output/e5_qdrant" \
#     --gte_qdrant_path "/home/fit02/dien-workspace/vlsp/output/gte_qdrant" \
#     --batch_size 104

# python3 -m ViDRILL.pipeline.corpus_encoder \
#     --corpus_path "/home/fit02/dien-workspace/vlsp/data_drill/maxlen500/corpus.csv"\
#     --bge_qdrant_path "/home/fit02/dien-workspace/vlsp/output/db_1stage_500_9000" \
#     --bge_model_name "/home/fit02/dien-workspace/vlsp/output/bge-m3-neg-top60-form-finetune-500/checkpoint-9000" \
#     --batch_size 104

python3 -m ViDRILL.prepare_data.hard_neg \
    --train_file "/home/fit02/dien-workspace/vlsp/data_drill/maxlen450/train_e5_450.csv" \
    --top_k_retrieval 60 \
    --bge_model_name "/home/fit02/dien-workspace/vlsp/output/bge-m3-neg-top60-form-finetune" \
    --output_file "/home/fit02/dien-workspace/vlsp/data_drill/maxlen450/train_neg-top60-form-finetune.json" \
    --bge_qdrant_path "/home/fit02/dien-workspace/vlsp/output/db_bge_450_from_bge_2048"
# python3 -m ViDRILL.prepare_data.choice_hard_neg \
#     --input_file "/home/fit02/dien-workspace/vlsp/data_drill/maxlen450/train_rerank_450_e5_gte.json" \
#     --output_file "/home/fit02/dien-workspace/vlsp/data_drill/maxlen450/train_rerank_450_e5_gte_start20_50.json" \
#     --start 20 \
#     --end 50
