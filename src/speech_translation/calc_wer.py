import evaluate
import os
import json
import zhconv
import re
import editdistance

from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer


def combine_json(out_dir, ref_file, out_file):
    sents = {}
    for line in open(ref_file):
        if len(line.split()) == 1:
            sid, tokens = line.strip(), ''
        else:
            sid, tokens = line.strip().split(None, 1)
        tokens = zhconv.convert(tokens, 'zh-cn')
        if sid.endswith('-hyp:'):
            sent = {}
            sent['hyp'] = tokens
        elif sid.endswith('-ref:'):
            sent['ref'] = tokens
            sid = sid[:-5]
            sents[sid] = sent

    data_list = []
    for jf in os.listdir(out_dir):
        sid = jf[:-5]
        if sid in sents:
            with open(os.path.join(out_dir, jf)) as file:
                try:
                    data = json.load(file)
                except:
                    print('Error:', os.path.join(out_dir, jf))
                data['sid'] = sid
                data['hypothesis'] = sents[sid]['hyp']
                data['reference'] = sents[sid]['ref']
                data_list.append(data)
    if len(data_list) != len(sents):
        print('Not completed!!')
    # assert len(data_list) == len(ref_dict)

    with open(out_file, 'w', encoding='utf8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)


def formatting(data_list, pid):
    std = EnglishTextNormalizer()
    useless_sents = [
        'Just translate the text.',
        'It is the same as in English:',
    ]
    err_cnt = 0
    for data in data_list:
        if pid == 'r4' or pid == 'p4':
            try:
                json_data = data['output'].split('}', 1)[0] + '}'
                trans = json.loads(json_data)['translation']
                if type(trans) == list:
                    trans = ' '.join(trans)
            except:
                json_match = re.findall(r'"([^"]*)"', data['output'])
                if json_match and len(json_match) >= 2 and json_match[0] == 'translation':
                    trans = json_match[1]
                else:
                    err_cnt += 1
                    trans = data['hypothesis']
                    # print(data['output'])
            data['output_std'] = trans
            # print(trans)
        elif pid == 'c4':
            try:
                json_data = data['output'].rsplit('}', 1)[0] + '}'
                json_data = '{' + json_data.split('{', 1)[1]
                cnts = json_data.count('{') - json_data.count('}')
                json_data = json_data + '}' * cnts
                # print(json_data)
                json_data = json.loads(json_data) #.split('}', 1)[0] + '}'
                if 'transcription' in json_data and 'corrected_transcription' in json_data['transcription']:
                    recog = json_data['transcription']['corrected_transcription']
                elif 'corrected_transcription' in json_data:
                    recog = json_data['corrected_transcription']
                if type(recog) == list:
                    recog = ' '.join(recog)
                elif type(recog) == dict:
                    recog = ' '.join(recog.values())
                # recog = data['hypothesis']
            except:
                # json_match = re.findall(r'"([^"]*)"', data['output'])
                # if json_match and len(json_match) >= 2 and json_match[0] == 'translation':
                #     trans = json_match[1]
                # else:
                err_cnt += 1
                recog = data['hypothesis']
                print(data['output'])
            data['output_std'] = recog
        else:
            data['output_std'] = data['output']
        for sent in useless_sents:
            data['output_std'] = data['output_std'].replace(sent, '')
        data['output_std'] = std(data['output_std'])
        data['output_std'] = zhconv.convert(data['output_std'], 'zh-cn')
        data['reference_std'] = std(data['reference'])
        data['reference_std'] = zhconv.convert(data['reference_std'], 'zh-cn')
    print('Errors:', err_cnt, err_cnt / len(data_list))


def calc_bleu(data_list, pid, out_file):
    bleu = evaluate.load('bleu')
    
    formatting(data_list, pid)
    with open(out_file, 'w', encoding='utf8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)

    predictions = [data['output_std'] for data in data_list]
    references = [[data['reference_std']] for data in data_list]

    eval_results = bleu.compute(predictions=predictions, references=references)
    print(eval_results["bleu"])


def calc_wer(data_list, pid, out_file, lang):
    errors, refs = 0, 0
    formatting(data_list, pid)
    with open(out_file, 'w', encoding='utf8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)

    for data in data_list:
        if lang == 'zh':
            errors += editdistance.eval(data['output_std'], data['reference_std'])
            refs += len(data['reference_std'])
        else:
            errors += editdistance.eval(data['output_std'].split(), data['reference_std'].split())
            refs += len(data['reference_std'].split())

    wer = errors / refs
    print(wer)


for pid in ['c4']:
    print(f'=========== {pid} ===========')
    for lang in ['et', 'lv', 'ru', 'zh']:
# for lang in ['lv']:
    # ref_path = f'covost/test_st/covost_v2_{lang}_list'
        ref_path = f'exp/baseline/large/covost/transcribe/False_{lang}_beam5_stampFalse_nonorm_100'
        out_dir = f'st_exp/mistral-7b/{pid}/{lang}'
        # out_dir = f'st_exp/gpt-3.5/{pid}/{lang}'
        pid = out_dir.split('/')[-2]
        # out_dir = f'st_exp/gpt-4/r3/{lang}'
        print(out_dir)

        file = out_dir + '_100.json'
        combine_json(out_dir, ref_path, file)
        data_list = json.load(open(file))
        calc_wer(data_list, pid, out_dir + '_100_std.json', lang)
