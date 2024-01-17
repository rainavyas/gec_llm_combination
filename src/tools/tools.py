import torch
import random
import spacy 

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def get_default_device(gpu_id=0):
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device(f'cuda:{gpu_id}')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def normalize_text(text):
    '''
    Add space before punctuation
    '''
    puncts = [".", ",", "?", ")", "'s"]
    for p in puncts:
        text = text.replace(p, ' '+p)
    text = text.replace("(", ' '+"(")
    return text

NLP = spacy.load("en_core_web_sm")
def spacy_normalize_text(text):
    '''
    Used in BEA-19 test set for generating input sentences
    '''
    doc = NLP(text)
    return ' '.join([t.text for t in doc])