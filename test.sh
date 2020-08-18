python test.py --task test \
        --batch_size 1 \
        --check_point "/workspace/pt1/log/reimp/ep18.pkl" \
        --eval_set test \
        --data_worker 8 \
        --video_path "/workspace/c3d_res_phoenix_body_iter5_120k" \
        --gpu 0 \
        --early_exit "3,3,3" \
        --decoder_learned_pos