'''
Returns the following files:

1) Manual fluent transcriptions
2) Manual GEC transcriptions
3) Whisper_fluent transcriptions
4) Whisper_gec GEC transcriptions
5) BART GEC transcriptions (of whisper_fluent)

All text is normalized to be lower-case with no punctuation (other than apostrophe)

Format is a sample per line:

text1
text2
.
.
.

Assuming all the files are aligned
'''
import spacy

nlp = spacy.load('en')
def format_text(text):
    return ' '.join([str(t) for t in nlp(text)])

if __name__ == '__main__':

    man_fluent_path = '/scratches/dialfs/alta/sb2549/whisper_data/phrase_based/tsvs/flt/test_flt.tsv'
    man_gec_path = '/scratches/dialfs/alta/sb2549/whisper_data/phrase_based/tsvs/gec/test_gec.tsv'
    whisper_fluent_path = '/scratches/dialfs/alta/rm2114/exp-linguaskill/whisper_mrx/exp/small.en/LNG_flt_phrase_v2/prompt0_lr1e-5_lower/transcribe/True_test_beam5_stampFalse_speech'
    whisper_gec_path = '/scratches/dialfs/alta/rm2114/exp-linguaskill/whisper_mrx/exp/small.en/LNG_gec_phrase_v2/prompt0_lr1e-5_lower/transcribe/True_test_beam5_stampFalse_speech'
    bart_gec_path = '/scratches/dialfs/alta/sb2549/whisper_data/phrase_based/tsvs/AUTOGEC/bart/pretrained+finetuned_in_domain/whisper_flt2autogec/test.tsv'

    # manual fluent
    with open(man_fluent_path, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]
    out = []
    for l in lines:
        parts = l.split()
        text = format_text(' '.join(parts[2:]).lower())+'\n'
        out.append(text)

    with open('../experiments/files/manual_fluent.txt', 'w') as f:
        f.writelines(out)
    

    # manual gec
    with open(man_gec_path, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]
    out = []
    for l in lines:
        parts = l.split()
        text = format_text(' '.join(parts[2:]).lower())+'\n'
        out.append(text)

    with open('../experiments/files/manual_gec.txt', 'w') as f:
        f.writelines(out)
    

    # whisper fluent
    with open(whisper_fluent_path, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]
    out = []
    for l in lines:
        parts = l.split()
        if parts[0][-2] == 'f':
            continue
        text = format_text(' '.join(parts[1:]).lower())+'\n'
        out.append(text)

    with open('../experiments/files/whisper_fluent.txt', 'w') as f:
        f.writelines(out) 


    # whisper gec
    with open(whisper_gec_path, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]
    out = []
    for l in lines:
        parts = l.split()
        if parts[0][-2] == 'f':
            continue
        text = format_text(' '.join(parts[1:]).lower())+'\n'
        out.append(text)

    with open('../experiments/files/whisper_gec.txt', 'w') as f:
        f.writelines(out) 

    # bart gec
    with open(bart_gec_path, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]
    out = []
    for l in lines:
        parts = l.split()
        text = format_text(' '.join(parts[1:]).lower())+'\n'
        out.append(text)

    with open('../experiments/files/bart_gec.txt', 'w') as f:
        f.writelines(out) 
        
