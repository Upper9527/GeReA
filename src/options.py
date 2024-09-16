import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=1000)
        self.parser.add_argument('--total_steps', type=int, default=1000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None, 
                        help='total number of step for the scheduler, if None then scheduler_total_step = total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=1)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adam')
        self.parser.add_argument('--scheduler', type=str, default='fixed')
        self.parser.add_argument('--weight_decay', type=float, default=0.1)
        self.parser.add_argument('--fixed_lr', action='store_true')

    def add_eval_options(self):
        self.parser.add_argument('--write_results', action='store_true', help='save results')

    def add_reader_options(self):
        self.parser.add_argument('--train_data', type=str, default='none', help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default='none', help='path of eval data')
        self.parser.add_argument('--num_workers', type=int, default=4, help='num workers')
        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the encoder')
        self.parser.add_argument('--box_number', type=int, default=36, help='number of detected objects')
        self.parser.add_argument('--text_maxlength', type=int, default=200, 
                        help='maximum number of tokens in text segments (question+passage)')
        self.parser.add_argument('--answer_maxlength', type=int, default=-1, 
                        help='maximum number of tokens used to train the model, no truncation if -1')
        self.parser.add_argument('--no_title', action='store_true', 
                        help='article titles not included in passages')
        self.parser.add_argument('--n_im_context', type=int, default=1, help='the number of implicit knowledge samples')
        self.parser.add_argument('--n_ex_context', type=int, default=1, help='the number of explicit knowledge samples')
        self.parser.add_argument('--n_tags', type=int, default=32, help='the number of tags')
        self.parser.add_argument('--n_block', type=int, default=8, help='the number of blocks')
        self.parser.add_argument('--t5_feat_dim', type=int, default=512)
        self.parser.add_argument('--t5_pos_dim', type=int, default=4)

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/', help='models are saved here')
        self.parser.add_argument('--model_path', type=str, default='none', help='path for retraining')

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=1, type=int, 
                        help="Batch size per GPU/CPU for training.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                        help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")
        # training parameters
        self.parser.add_argument('--device', type=str, default="cuda", help='which device the training is on.')
        self.parser.add_argument('--print_freq', type=int, default=100,
                        help='print loss every <print_freq> steps during training')
        self.parser.add_argument('--eval_freq', type=int, default=1000,
                        help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--save_freq', type=int, default=2500,
                        help='save model every <save_freq> steps during training')
        self.parser.add_argument('--eval_print_freq', type=int, default=1000,
                        help='print intermdiate results of evaluation every <eval_print_freq> steps')
        #self.parser.add_argument('--write_results', action='store_true', help='save results')


    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir)/ opt.name
        model_dir = expr_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir/'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        return opt


def get_options(use_reader=False,
                use_retriever=False,
                use_optim=False,
                use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()
