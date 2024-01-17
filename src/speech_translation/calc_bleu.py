import evaluate
import os
import json
import zhconv

from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer


def combine_json(out_dir, ref_file):
    ref_dict = {}
    for line in open(ref_file):
        sid, _, sent = line.split(None, 2)
        sent = zhconv.convert(sent, 'zh-cn')
        ref_dict[sid] = sent

    data_list = []
    for jf in os.listdir(out_dir):
        sid = jf[:-5]
        with open(os.path.join(out_dir, jf)) as file:
            try:
                data = json.load(file)
            except:
                print('Error:', os.path.join(out_dir, jf))
            data['sid'] = sid
            data['reference'] = ref_dict[sid]
            data_list.append(data)
    assert len(data_list) == len(ref_dict)

    with open(out_dir + '.json', 'w', encoding='utf8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)


def formatting(data_list):
    std = EnglishTextNormalizer()
    useless_sents = [
        'Just translate the text.',
        'It is the same as in English:',
    ]
    for data in data_list:
        data['output_std'] = data['output']
        for sent in useless_sents:
            data['output_std'] = data['output_std'].replace(sent, '')
        data['output_std'] = std(data['output_std'])
        data['reference_std'] = std(data['reference'])


def calc_bleu(data_list):
    bleu = evaluate.load('bleu')
    
    formatting(data_list)
    with open(out_dir + '_std.json', 'w', encoding='utf8') as file:
        json.dump(data_list, file, ensure_ascii=False, indent=2)

    predictions = [data['output_std'] for data in data_list]
    references = [[data['reference_std']] for data in data_list]

    eval_results = bleu.compute(predictions=predictions, references=references)
    print(eval_results["bleu"])


for lang in ['et', 'ru', 'zh']:
    ref_path = f'covost/test_st/covost_v2_{lang}_list'
    out_dir = f'st_exp/mistral-7b/r3/{lang}'
    print(out_dir)
    combine_json(out_dir, ref_path)

    with open(os.path.join(out_dir + '.json')) as file:
        data_list = json.load(file)
        calc_bleu(data_list)
