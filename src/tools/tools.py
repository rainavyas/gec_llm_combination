import torch
import random
import spacy 
from nltk.tokenize import word_tokenize

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

def normalize_text(text, data_name='conll'):
    '''
    Tokenization to be consistent with input for GEC evaluation
    '''
    if data_name == 'conll':
        return nltk_normalize_text(text)
    elif data_name == 'fce' or data_name == 'bea':
        return spacy_normalize_text(text)

NLP = spacy.load("en_core_web_sm")
def spacy_normalize_text(text):
    '''
    Used in BEA-19 test set for generating input sentences
    '''
    doc = NLP(text)
    return ' '.join([t.text for t in doc])

def nltk_normalize_text(text):
    return ' '.join(word_tokenize(text))


def spoken_gec_normalize_text(text):
    '''
    Lower case everything and remove all punctuation
    apply spacy tokenization
    '''
    punc = '''!:;\,./?'''
    res = ""
    text = text.lower()
    for ele in text:
        if ele not in punc:
            res+=ele
    return spacy_normalize_text(res) 