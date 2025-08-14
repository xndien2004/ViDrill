export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running pipeline script..."

CUDA_VISIBLE_DEVICES=0

python3 -m ViDRILL.eval.eval \
    --true_path /home/fit02/dien-workspace/vlsp/data_drill/origin/train_orgin.json \
    --pred_path /home/fit02/dien-workspace/vlsp/results.json