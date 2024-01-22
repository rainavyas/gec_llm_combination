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
    parser.add_argument('--combination', type=str, choices=['mbr', 'maxvote', 'llm-selection-mistral-7b', 'llm-selection-llama-7b', 'rand', 'llm-combination-mistral-7b'], default='mbr', help='method of combination.')
    parser.add_argument('--pred_files', type=str, required=True, nargs='+', help='path to outputs with predicted sequences. If spoken, pass files in order: disfluent, fluent, gec')
    parser.add_argument('--input_file', type=str, default='no-file', help='path to input file with source incorrect sequences')
    parser.add_argument('--outfile', type=str, required=True, help='path to save final predictions after combination')
    parser.add_argument('--votes', type=int, default=2, help='number of model votes to accept an edit for max voting combination')
    parser.add_argument('--gpu_id', type=int, default=0, help='select specific gpu')
    parser.add_argument('--template', type=int, default=2, help='prompt template for combination')
    parser.add_argument('--spoken', action='store_true', help='Perform spoken GEC combination')
    return parser.parse_known_args()

    