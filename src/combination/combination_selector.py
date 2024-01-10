from src.utils.helpers import read_lines
from .max_vote import Maxvotecombiner
from .mbr import MBRcombiner


def combination_selector(args):
    source_sentences = read_lines(args.input_file)

    pred_texts = []
    for pred_path in args.pred_files:
        pred_texts.append(read_lines(pred_path))
    
    if args.combination == 'maxvote':
        return Maxvotecombiner(source_sentences, pred_texts, min_count=args.votes)
    elif args.combination == 'mbr':
        return MBRcombiner(source_sentences, pred_texts)
    
