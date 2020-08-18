python train.py \
	--task train_conv_5 \
	--log_dir ./log/reimp-conv \
	--learning_rate 1e-3 \
    --weight_decay 1e-4 \
	--data_worker 8 \
    --print_step 500 \
	--video_path "/workspace/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px" \
	--gpu 0 \
    --check_point "/workspace/pt1/log/reimp/.pkl" \
    --batch_size 2 \
    --stage_epoch 100 \
#     --DEBUG True
