import argparse

class Args_b:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        # args for path
        
        parser.add_argument('--data_dir', default='./data/origin',
                            help='the ernie data dir')

        parser.add_argument('--output_dir', default='../model_crf/ernie_crf_b',
                            help='the output dir for model checkpoints')
        
        parser.add_argument('--save_dir', default='./save/dialogue/',
                            help='ernie_save')

        parser.add_argument('--bert_dir', default='../model_crf/chinese_roberta_wwm_ext_pytorch',
                            help='bert dir for ernie / roberta-wwm / uer')

        parser.add_argument('--bert_type', default='roberta_wwm',
                            help='roberta_wwm / ernie_1 / uer_large')
        
        parser.add_argument('--from_pretrained', default='ernie-1.0',
                            help='ernie model')

        parser.add_argument('--task_type', default='crf',
                            help='crf / span / mrc')

        parser.add_argument('--loss_type', default='ls_ce',
                            help='loss type for span bert')

        parser.add_argument('--use_type_embed', default=False, action='store_true',
                            help='weather to use soft label in span loss')

        parser.add_argument('--use_fp16', default=False, action='store_true',
                            help='weather to use fp16 during training')

        # other args
        parser.add_argument('--seed', type=int, default=123, help='random seed')

        parser.add_argument('--gpu_ids', type=str, default='0,1',
                            help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

        parser.add_argument('--mode', type=str, default='train',
                            help='train / stack')

        parser.add_argument('--max_seq_len', default=128, type=int)
        
        parser.add_argument('--max_steps', default=2000, type=int)

        parser.add_argument('--eval_batch_size', default=32, type=int)

        parser.add_argument('--swa_start', default=3, type=int,
                            help='the epoch when swa start')
        parser.add_argument('--no', default=14, type=int)

        # train args
        parser.add_argument('--train_epochs', default=5, type=int,
                            help='Max training epoch')

        parser.add_argument('--dropout_prob', default=0.1, type=float,
                            help='drop out probability')

        parser.add_argument('--lr', default=2e-5, type=float,
                            help='learning rate for the bert module')

        parser.add_argument('--other_lr', default=2e-3, type=float,
                            help='learning rate for the module except bert')

        parser.add_argument('--max_grad_norm', default=1.0, type=float,
                            help='max grad clip')

        parser.add_argument('--warmup_proportion', default=0.1, type=float)

        parser.add_argument('--weight_decay', default=0.00, type=float)

        parser.add_argument('--adam_epsilon', default=1e-8, type=float)

        parser.add_argument('--train_batch_size', default=32, type=int)

        parser.add_argument('--eval_model', default=True, action='store_true',
                            help='whether to eval model after training')

        parser.add_argument('--attack_train', default='', type=str,
                            help='fgm / pgd attack train when training')

        # test args
        parser.add_argument('--version', default='v0', type=str,
                            help='submit version')

        parser.add_argument('--submit_dir', default='submit', type=str)

        parser.add_argument('--ckpt_no', default='1', type=str)

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()
