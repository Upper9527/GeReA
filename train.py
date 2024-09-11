import pdb
import time
import sys
import torch
import transformers
from transformers import T5Config
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
#import dist
#dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=5400))
def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    device = opt.device
    model.train()
    while step < opt.total_steps:
        for _, batch in enumerate(train_dataloader):
            (idx, labels, _, context_ids, context_mask, vis_feats) = batch
            train_loss = model(
                vis_inputs=vis_feats.to(device),
                input_ids=context_ids.to(device),
                attention_mask=context_mask.to(device),
                labels=labels.to(device)
            )[0]

            train_loss.backward()
            loss_value = train_loss.item()
            if opt.local_rank == 0 and step % opt.print_freq == 0:
                print('Train Step : [{}/{}]  Train Loss: {:.3f}'.format(step, opt.total_steps, loss_value))

            step+= 1
            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if step % opt.eval_freq == 0:
                if opt.local_rank ==0:
                    print('evaluating !!!')
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps}"
                    log += f" train: {curr_loss/opt.eval_freq:.3f}"
                    log += f" accuracy: {100*dev_em:.2f}%"
                    log += f" lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)    
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")

            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        num_workers=opt.num_workers,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        pin_memory=True,
        collate_fn=collator
    )
    model.eval()
    total, device = 0, opt.device
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (idx, _, _, context_ids, context_mask, vis_feats) = batch
            outputs = model.generate(
                vis_inputs=vis_feats.to(device),
                input_ids=context_ids.to(device),
                attention_mask=context_mask.to(device),
                max_length=10
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers_list']
                score = src.evaluation.ems(opt, ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data,
        split_type="train2014",
        global_rank=opt.global_rank, 
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(opt, train_examples)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        split_type="val2014",
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(opt, eval_examples)

    t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
    t5_config = T5Config.from_pretrained(model_name)
    t5_config.feat_dim, t5_config.pos_dim = opt.t5_feat_dim, opt.t5_pos_dim

    model = src.model.FiDT5(t5_config, opt)
    model.load_t5(t5.state_dict())
    model = model.to(opt.local_rank)
    optimizer, scheduler = src.util.set_optim(opt, model)
    step, best_dev_em = 0, 0.0

    model.set_checkpoint(opt.use_checkpoint)
    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
        )
