import os
import pdb
import torch
import transformers
from transformers import T5Config
import numpy as np
from pathlib import Path
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model
from converter.convert import convert

def evaluate(model, dataset, dataloader, tokenizer, opt):
    loss, curr_loss = 0.0, 0.0
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total, device = 0, opt.device
    exactmatch = []
    if opt.write_results:
        write_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        fw = open(write_path / ('%d.txt'%opt.global_rank), 'a')
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
                example = dataset.data[idx[k]]
                if 'answers_list' in example:
                    score = src.evaluation.ems(opt, ans, example['answers_list'], i, len(dataset))
                    exactmatch.append(score)

                if opt.write_results:
                    fw.write(str(example['id']) + "\t" + ans + '\n')

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f'Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}'
                if len(exactmatch) == 0:
                    log += '| no answer to compute scores'
                else:
                    log += f' | average = {np.mean(exactmatch):.3f}'
                logger.warning(log)

    logger.warning(f'Process rank: {opt.global_rank}, total: {total} test examples | accuracy = {100*np.mean(exactmatch):.2f}%')
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    
    return score, total


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir)/opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / 'test_results').mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / 'run.log')
    if not directory_exists and opt.is_main:
        options.print_options(opt)
    
    model_name = 't5-' + opt.model_size
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    collator_function = src.data.Collator(opt.text_maxlength, tokenizer, answer_maxlength=opt.answer_maxlength)
    eval_examples = src.data.load_data(
        opt.eval_data,
        split_type="val2014",
        global_rank=0, #use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=1
    )
    eval_dataset = src.data.Dataset(
        opt,
        eval_examples, 
    )

    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=opt.per_gpu_batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collator_function
    )
    
    t5_config = T5Config.from_pretrained(model_name)
    t5_config.feat_dim, t5_config.pos_dim = opt.t5_feat_dim, opt.t5_pos_dim

    ori_model = src.model.FiDT5(t5_config, opt)
    print('model path: ', opt.model_path)
    model = ori_model.from_pretrained(opt.model_path)
    model = model.to(opt.device)

    logger.info("Start eval")
    exactmatch, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f'Accuracy: {100*exactmatch:.2f}%, total number of example: {total}')

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / 'test_results'
        write_path = Path(opt.checkpoint_dir) / opt.name / 'final_output.txt'
        src.util.write_output(glob_path, write_path) 
        convert(write_path, os.path.join(opt.checkpoint_dir, opt.name))
        

