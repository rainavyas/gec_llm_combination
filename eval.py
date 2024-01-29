'''
WER and TER evaluation
'''

import sys
import os
import argparse

from src.tools.tools import eval_wer, eval_ter
from src.utils.helpers import read_lines

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--hyp', type=str, help='Path to hypothesis')
    commandLineParser.add_argument('--ref', type=str, help='Path to referece')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    hyps = read_lines(args.hyp)
    ref = read_lines(args.ref)
    wer = eval_wer(hyps, ref)
    ter = eval_ter(hyps, ref)

    print("WER (%):", wer*100)
    print("TER (%):", ter*100)
    

