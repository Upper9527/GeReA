NGPU=2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 10845 train.py \
	--use_checkpoint \
	--lr 5e-5 \
	--model_size large \
	--num_workers 8 \
	--optim adamw \
	--scheduler linear \
	--weight_decay 0.01 \
	--save_freq 10000 \
	--eval_freq 10000 \
	--print_freq 100 \
	--text_maxlength 400 \
	--seed 833 \
	--name exp \
	--checkpoint_dir ./checkpoints_InstructBLIP+LLaVA-1.5 \
	--per_gpu_batch_size 1 \
	--total_step 20000 \
	--warmup_step 1000 

