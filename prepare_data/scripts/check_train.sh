export PYTHONPATH="/home/fit02/dien-workspace/vlsp/ViDRILL:$PYTHONPATH"
echo "Running training script..."

CUDA_VISIBLE_DEVICES=0
python3 -m ViDRILL.prepare_data.check_train