'''
Takes in 3 files:
1) .inc 2) predictions.txt 3) .corr

where each file is of the format:
ID1 TEXT1
ID2 TEXT2
.
.
.

Outputs 3 files with IDs aligned:
1) base_name.inc 2) base_name.pred 3) base_name.corr

where each file is of the format:
TEXT1
TEXT2
.
.
.


Also normalizes the input, prediction and correction texts to be consistent in tokenization
'''

import sys
import os
import argparse
from tqdm import tqdm

from src.tools.tools import spacy_normalize_text, spoken_gec_normalize_text

def get_sentences_dict(data_path, normalize=False, s_norm=False):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]
    id2text = {}
    for l in tqdm(lines):
        parts = l.split()
        id = parts[0]
        text = ' '.join(parts[1:])
        if normalize:
            text = spacy_normalize_text(text)
        if s_norm:
            text = spoken_gec_normalize_text(text)
        id2text[id] = text
    return id2text

def align_data(inc_dict, pred_dict, corr_dict):
    inc_sens = []
    pred_sens = []
    corr_sens = []
    for i, (id, text) in enumerate(corr_dict.items()):
        try:
            pred_sens.append(pred_dict[id]+'\n')
            inc_sens.append(inc_dict[id]+'\n')
            corr_sens.append(text+'\n')
        except:
            # print(f'{i}) {id} in corrected but not in predicted')
            pass
    assert len(pred_sens) == len(inc_sens), "Mismatch in num items"
    return inc_sens, pred_sens, corr_sens

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('INC', type=str, help='Path to incorrect test data')
    commandLineParser.add_argument('PRED', type=str, help='Path to output predicted data')
    commandLineParser.add_argument('CORR', type=str, help='Path to correct output test data')
    commandLineParser.add_argument('BASE', type=str, help='Path base for output files')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/align.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Get sentences and align
    normalize=False
    if 'fce' in args.INC:
        normalize=True
    s_norm = False
    if 'spoken' in args.INC:
        s_norm = True
    inc_id2text = get_sentences_dict(args.INC, normalize=normalize, s_norm=s_norm)
    pred_id2text = get_sentences_dict(args.PRED, normalize=normalize, s_norm=s_norm)
    corr_id2text = get_sentences_dict(args.CORR, normalize=normalize, s_norm=s_norm)
    inc_sens, pred_sens, corr_sens = align_data(inc_id2text, pred_id2text, corr_id2text)

    # Save to output files
    filename = f'{args.BASE}.inc'
    with open(filename, 'w') as f:
        f.writelines(inc_sens)
    filename = f'{args.BASE}.pred'
    with open(filename, 'w') as f:
        f.writelines(pred_sens)
    filename = f'{args.BASE}.corr'
    with open(filename, 'w') as f:
        f.writelines(corr_sens)

