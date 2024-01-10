''' Prediction after blackbox combination of multiple model outputs'''

import sys
import os

from src.tools.args import combined_args
from src.combination.combination_selector import combination_selector

if __name__ == "__main__":

    # get command line arguments
    args, c = combined_args()
    print(args)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict_combined.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Perform combination
    combiner = combination_selector(args)
    combined_texts = combiner.combined_texts

    # save the predictions
    with open(args.outfile, 'w') as f:
        f.write("\n".join(combined_texts) + '\n')