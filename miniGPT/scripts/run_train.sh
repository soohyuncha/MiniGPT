python main.py \
    --dataset_name WikiText2 \
    --dataset_mode preprocess \
    --train_mode train \
    --train_epochs 20 \
    --gen_mode eval_set \
    --model_path './model/miniGPT.pt' \
    \
    --batch_size 8 \
    --chunk_size 200 \
    \
    --d_model 768 \
    --d_hidden 1536 \
    --n_layer 12 \
    --n_head 12 \
