import torch
import torch.nn.functional as F
import sys
import pandas
import openai
import argparse
import os
import json
import zhconv
import random
import time

from types import SimpleNamespace
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from general import check_output_path, save_script_args, check_output_dir, LANG_CODE

LLAMA_MAX_LEN = 3800

MODEL_URLS = {
    'llama2-7b':'meta-llama/Llama-2-7b-hf',
    'llama2-13b':'meta-llama/Llama-2-13b-hf',
    'llama2-7b-chat':'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b-chat':'meta-llama/Llama-2-13b-chat-hf',
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
}

prompt_list = {
    'p1': "Please translate the following text from {} into English: <text> {} </text>. Do not add any explanation.",
    'p2': "Please translate the following text from {} into English: {}. Do not add any explanation.",
    'p3': "Please translate the following text from {} into English: {}\n\nDo not add any explanation.",
    'r3': "Please translate the following text from {} into English: {}\n\nDo not add any explanation.",
    'c4': "For a given {} utterance, the transcription produced by a speech recognition system is: {}\n\nThe output from an end-to-end speech translation system for the same utterance is: {}\n\nBoth sentences might have errors made by the system. Consider the phonetic similarity to the speech recognition result and the semantic meaning in the speech translation. Generate a corrected speech recognition result in {} and an improved English translation, providing them in XML tags of <transription> </transcription> and <translation> </translation> in two lines without explanations. For the corrected ASR result, try to be as close as possible to the original ASR output."
}


def load_sents(path):
    sents = []
    for line in open(path):
        if len(line.split()) == 1:
            sid, tokens = line.strip(), ''
        else:
            sid, tokens = line.strip().split(None, 1)
        tokens = zhconv.convert(tokens, 'zh-cn')
        if sid.endswith('-hyp:'):
            sent = {}
            sent['sid'] = sid[:-5]
            sent['hyp'] = tokens
        elif sid.endswith('-ref:'):
            sent['ref'] = tokens
            sents.append(sent)
    return sents


class GPTInterface:
    def __init__(self, system):
        self.llm_model = system

    def text_response(self, input_text):
        # ChatGPT
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ],
            temperature=0.0, # 0.0 = deterministic
            max_tokens=1000, # max_tokens is the generated one,
        )
        gen_text = response.choices[0].message.content
        return gen_text


class Llm2Interface:
    def __init__(self, system, device=None):
        system_url = MODEL_URLS[system]
        self.tokenizer = AutoTokenizer.from_pretrained(system_url)
        self.model = AutoModelForCausalLM.from_pretrained(system_url)
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            self.model = self.model.half()
        self.to(device)
        self.device = device

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def text_response(self, input_text, top_k:int=10, do_sample:bool=False, max_new_tokens:int=None):
        # inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=LLAMA_MAX_LEN).to(self.device)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        output = self.model.generate(
            input_ids=inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            top_k=top_k,
            do_sample=do_sample,
            max_new_tokens=1000,
            pad_token_id=self.tokenizer.eos_token_id
        )

        output_tokens = output[0]

        input_tokens = inputs.input_ids[0]
        new_tokens = output_tokens[len(input_tokens):]
        assert torch.equal(output_tokens[:len(input_tokens)], input_tokens)

        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return output_text


def add_arguments(parser):
    '''Build Argument Parser'''
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--llm_model', type=str, default='mistral-7b')
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--lang', type=str, default='lv')
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--pid', type=str, default='p3')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--shuffle', type="bool", nargs="?", const=True, default=False)
    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = args.lang
    out_dir = f"st_exp/{args.llm_model}/{args.pid}/{args.output_dir}"
    check_output_dir(out_dir)

    if 'gpt' in args.llm_model:
        if args.llm_model == 'gpt-3.5':
            args.llm_model = 'gpt-3.5-turbo-1106'
        llm = GPTInterface(system=args.llm_model)
    else:
        llm = Llm2Interface(system=args.llm_model)

    prompt = prompt_list[args.pid]
    if not args.input_file:
        args.input_file = f'exp/baseline/large/covost/transcribe/False_{args.lang}_beam5_stampFalse_nonorm'
        args.trans_file = f'exp/baseline/large/covost/translate/False_{args.lang}_beam5_stampFalse_nonorm'
    sents = load_sents(args.input_file)
    
    if args.debug:
        sents = sents[:3]
    if args.shuffle:
        random.shuffle(sents)

    for sent in sents:
        out_file = f"{out_dir}/{sent['sid']}.json"
        if os.path.exists(out_file):
            print(f"============= {sent['sid']} exists =============")
            continue

        if args.pid == 'p1' or args.pid == 'p2' or args.pid == 'p3':
            input = prompt.format(LANG_CODE[args.lang], sent['hyp'].strip())
        elif args.pid == 'r3':
            input = prompt.format(LANG_CODE[args.lang], sent['ref'].strip())
        elif args.pid == 'c4':
            input = prompt.format(LANG_CODE[args.lang], sent['hyp'].strip())

        print(input)
        output = llm.text_response(input)

        with open(out_file, 'w', encoding='utf8') as file:
            json_data = {}
            json_data['input'] = input
            json_data['output'] = output
            json.dump(json_data, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    for counter in range(1, 100):
        try:
            main()
        except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError):
            print("openai.error.RateLimitError... #{}".format(counter))
            print("restart in 5 seconds")
            time.sleep(5)
