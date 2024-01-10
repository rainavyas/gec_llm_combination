import sys
import os

from src.tools.args import core_args
from src.tools.tools import set_seeds
from src.data.load_data import load_data
from src.inference.model_selector import select_model

if __name__ == "__main__":

    # get command line arguments
    args, c = core_args()
    print(args)

    set_seeds(args.seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/predict.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # load inc data
    data, _ = load_data(args.data_name)

    # load model
    model = select_model(args)

    # get predictions
    outputs = model.predict(data)

    # save the predictions
    with open(args.output_file, 'w') as f:
        f.write("\n".join(outputs) + '\n')