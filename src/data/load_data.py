import json
from src.utils.helpers import read_lines

def load_data(data_name):
    with open('src/data/filepaths.json', 'r') as f:
        fpaths = json.load(f)
    fpaths = fpaths[data_name]

    # inputs
    inc_data = read_lines(fpaths['inc'])

    # outputs
    try:
        corr_data = read_lines(fpaths['corr'])
    except:
        corr_data = []
    
    return inc_data, corr_data