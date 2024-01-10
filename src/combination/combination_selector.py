from src.utils.helpers import read_lines
from .max_vote import Maxvotecombiner
from .mbr import MBRcombiner
from .llm_combination.combination import SelectionLLMCombiner


def combination_selector(args):
    source_sentences = read_lines(args.input_file)

    pred_texts = []
    for pred_path in args.pred_files:
        pred_texts.append(read_lines(pred_path))
    
    if args.combination == 'maxvote':
        return Maxvotecombiner(source_sentences, pred_texts, min_count=args.votes)
    elif args.combination == 'mbr':
        return MBRcombiner(source_sentences, pred_texts)
    elif 'llm' in args.combination:
        combination = args.combination
        if 'selection' in combination:
            comb_model_name = '-'.join(combination.split('-')[2:])
            return SelectionLLMCombiner(source_sentences, pred_texts, comb_model_name=comb_model_name)

    
