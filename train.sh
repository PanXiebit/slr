python train.py \
	--task train-5-2 \
	--log_dir ./log/reimp \
	--learning_rate 1e-4 \
	--data_worker 8 \
  --print_step 50 \
	--video_path "/workspace/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px" \
	--gpu 0 \
  --decoder_learned_pos \
  --check_point "/workspace/pt1/log/reimp/ep11_35.3000.pkl" \
  --batch_size 20 \
  --early_exit "6,6,6" \
  --stage_epoch 30 \
  --ctc_weight 0.1 \
  --dec_weight 1.0 \
  --train_cnn_in_decoder True \
  --no_share_maskpredictor \
  --no_share_discriminator \
#   --DEBUG True
