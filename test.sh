CUDA_VISIBLE_DEVICES=2 python test.py --eval_data processed_data/test.pkl \
	--model_size large \
	--per_gpu_batch_size 1 \
	--num_workers 8 \
	--text_maxlength 300 \
	--checkpoint_dir ./checkpoint_65.41/ \
	--seed 833 \
	--name eval \
	--model_path checkpoints_two_card_150/exp/checkpoint/best_dev-65.41/ \
    --write_results
