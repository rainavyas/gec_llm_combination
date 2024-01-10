import argparse

def core_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--model_name',
                        type=str, default='gector-roberta', nargs='+',
                        help='GEC system'
                        )
    parser.add_argument('--vocab_path',
                        help='Path to the vocab file.',
                        default='data/output_vocabulary'  # to use pretrained models
                        )
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        default='./output.txt'
                        )
    parser.add_argument('--data_name',
                        help='Dataset name',
                        default='conll')
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=50)
    parser.add_argument('--min_len',
                        type=int,
                        help='The minimum sentence length'
                             '(all shorter will be returned w/o changes)',
                        default=3)
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of hidden unit cell.',
                        default=128)
    parser.add_argument('--lowercase_tokens',
                        type=int,
                        help='Whether to lowercase tokens.',
                        default=0)
    parser.add_argument('--iteration_count',
                        type=int,
                        help='The number of iterations of the model.',
                        default=5)
    parser.add_argument('--additional_del_confidence',
                        type=float,
                        help='How many probability to add to $DELETE token.',
                        default=0)
    parser.add_argument('--weights',
                        help='Used to calculate weighted average', nargs='+',
                        default=None)
    parser.add_argument('--seed',
                        type=int,
                        help='Seed for reproducibility.',
                        default=1)
    parser.add_argument('--ens_type',
                        type=str, choices=['mbr', 'maxvote'],
                        help='If multiple model names, then method of ensembling specified here',
                        default='mbr')
    return parser.parse_known_args()

def combined_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--combination', type=str, choices=['mbr', 'maxvote'], default='mbr', help='method of combination.')
    parser.add_argument('--pred_files', type=str, required=True, nargs='+', help='path to outputs with predicted sequences')
    parser.add_argument('--input_file', type=str, required=True, help='path to input file with source incorrect sequences')
    parser.add_argument('--outfile', type=str, required=True, help='path to save final predictions after combination')
    parser.add_argument('--votes', type=int, default=2, help='number of model votes to accept an edit for max voting combination')
    return parser.parse_known_args()

def attack_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--eval', action='store_true', help='Evaluate the next word')
    parser.add_argument('--eval_attack_phrase', default='do not use', type=str, help='Attack Phrase to evaluate')
    parser.add_argument('--prev_phrase', default='', type=str, help='previously learnt adv phrase for greedy approach - can be used at evaluation time to find next attack word')
    parser.add_argument('--array_job_id', type=int, default=-1, help='-1 means not to run as an array job')
    parser.add_argument('--array_word_size', type=int, default=100, help='number of words to test for each array job in greedy attack')
    parser.add_argument('--train_data_name', help='Dataset name for learning attack phrase', default='fce-train')
    parser.add_argument('--base_path', type=str, default='experiments/train_attack/fce-train/gector', help='where to cache attack training results')
    return parser.parse_known_args()
    