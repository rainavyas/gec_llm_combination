from src.utils.helpers import read_lines
from .max_vote import Maxvotecombiner
from .mbr import MBRcombiner
from .llm_combination.combination import SelectionLLMCombiner, CombinationLLMCombiner
from .llm_combination.spoken_combination import SpokenLLMCombiner
from .random import RandCombiner


def combination_selector(args):
    pred_texts = []
    for pred_path in args.pred_files:
        pred_texts.append(read_lines(pred_path))
    
    if args.spoken:
        return SpokenLLMCombiner(pred_texts, comb_model_name=comb_model_name, gpu_id=args.gpu_id, template=args.template)
    else:
        source_sentences = read_lines(args.input_file)
    
        if args.combination == 'maxvote':
            return Maxvotecombiner(source_sentences, pred_texts, min_count=args.votes)
        elif args.combination == 'mbr':
            return MBRcombiner(source_sentences, pred_texts)
        elif 'llm' in args.combination:
            combination = args.combination
            comb_model_name = '-'.join(combination.split('-')[2:])
            if 'selection' in combination:
                return SelectionLLMCombiner(source_sentences, pred_texts, comb_model_name=comb_model_name, gpu_id=args.gpu_id, template=args.template)
            elif 'combination' in combination:
                dname = _get_dname(args.input_file)
                return CombinationLLMCombiner(source_sentences, pred_texts, comb_model_name=comb_model_name, gpu_id=args.gpu_id, template=args.template, dname=dname)
        elif args.combination == 'rand':
            return RandCombiner(pred_texts)

def _get_dname(filename):
    if 'bea' in filename or 'BEA' in filename:
        return 'bea'
    elif 'fce' in filename or 'FCE' in filename:
        return 'fce' 
    elif 'conll' in filename or 'CoNLL' in filename:
        return 'conll' 
    else:
        raise ValueError("Failed to get data name - necessary for text normalization")
    
