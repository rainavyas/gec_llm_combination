'''
Using LLM to combine model outputs
'''
from abc import ABC, abstractmethod
import re

from .select_llm_base_model import get_model
from src.tools.tools import normalize_text

class BaseLLMCombiner(ABC):
    def __init__(self, source_sentences, pred_texts, comb_model_name='mistral-7b', gpu_id=0, template=2):
        self.template = template
        self.comb_model = get_model(comb_model_name, gpu_id=gpu_id)
        self.combined_texts = self._make_all_changes(source_sentences, pred_texts)
    
    
    def _make_all_changes(self, source_sentences, pred_texts):
        prompts = self._prep_prompts(source_sentences, pred_texts)
        return self.comb_model.predict_all(prompts)
    
    @abstractmethod
    def _prep_prompts(self, source_sentences, pred_texts):
        pass


class SelectionLLMCombiner(BaseLLMCombiner):
    def __init__(self, *args, **kwargs):
        BaseLLMCombiner.__init__(self, *args, **kwargs)

    def _make_all_changes(self, source_sentences, pred_texts):
        preds = list(zip(*pred_texts)) # outer iter over samples not models
        prompts = self._prep_prompts(source_sentences, preds)
        selections = self.comb_model.predict_all(prompts)

        # extract selected option
        predictions = []
        for sel, ps in zip(selections, preds):
            try:
                result = re.search('<option>(.*)</option>', sel)
                num = int(result.group(1)) - 1
            except:
                num = 0
                print('Output format failed for LLM')
            predictions.append(ps[num])
        return predictions

    def _prep_prompts(self, source_sentences, preds):
        prompts = [self._prompt(s, ps) for s,ps in zip(source_sentences, preds)]
        return prompts
    
    def _prompt(self, source, preds, remove_ids_in_prompt=True):
        if remove_ids_in_prompt:
            source = ' '.join(source.split(' ')[1:])
        
        if self.template == 2:
            # 2-shot
            out = (
                "Select the best output sentence option for grammatical error correction "
                "of the given input sentence. Select only one output sentence option from {1,2,3} and return only the option number in the following format <option>1/2/3</option>.\n"
                "Here is an example.\n"
                "Input: The boy walk down street.\n"
                "option 1: The boy walks down street.\n"
                "option 2: The boy walking down street.\n"
                "option 3: The boy is walking down the street.\n"
                "Output: <option>3</option>\n\n"
                "Here is another example.\n"
                "Input: It was very difficult walk.\n"
                "option 1: It was very difficult walk.\n"
                "option 2: It was very difficult to walk.\n"
                "option 3: It was very difficult walking.\n"
                "Output: <option>2</option>\n\n"
                "Now select the best option for the following.\n"
                f"Input: {source}\n"
            )
            for i, pred in enumerate(preds):
                if remove_ids_in_prompt:
                    pred = ' '.join(pred.split(' ')[1:])
                out += f"option {i+1}: {pred}\n"
            out += 'Output:'

        if self.template == 3:
            # 2-shot
            out = (
                "Select the best output sentence option for grammatical error correction "
                "of the given input sentence. Select only one output sentence option from {1,2,3} and return only the option number in the following format <option>1/2/3</option>.\n"
                "Here is an example.\n"
                "Input: The boy walk down street.\n"
                "option 1: The boy walks down street.\n"
                "option 2: The boy walking down street.\n"
                "option 3: The boy is walking down the street.\n"
                "Output: <option>3</option>\n\n"
                "Here is another example.\n"
                "Input: Hello, how you?\n"
                "option 1: Hello, how are you?\n"
                "option 2: Hello, how you?\n"
                "option 3: Hello how are you.\n"
                "Output: <option>1</option>\n\n"
                "Here is another example.\n"
                "Input: It was very difficult walk.\n"
                "option 1: It was very difficult walk.\n"
                "option 2: It was very difficult to walk.\n"
                "option 3: It was very difficult walking.\n"
                "Output: <option>2</option>\n\n"
                "Now select the best option for the following.\n"
                f"Input: {source}\n"
            )
            for i, pred in enumerate(preds):
                if remove_ids_in_prompt:
                    pred = ' '.join(pred.split(' ')[1:])
                out += f"option {i+1}: {pred}\n"
            out += 'Output:'

        elif self.template == 0:
            # zero-shot
            out = (
                "Select the best output sentence option for grammatical error correction "
                "of the given input sentence. Select only one output sentence option from {1,2,3} and return only the option number in the following format <option>1/2/3</option>.\n"
                
                "Slect the best option for the following.\n"
                f"Input: {source}\n"
            )
            for i, pred in enumerate(preds):
                if remove_ids_in_prompt:
                    pred = ' '.join(pred.split(' ')[1:])
                out += f"option{i+1}: {pred}\n\n"
            out += 'Make sure to return in the format <option>1/2/3</option>. If option1 is the best return <option>1</option>. If option2 is the best return <option>2</option>. If option3 is the best return <option>3</option>.\n' 
        return out


class CombinationLLMCombiner(BaseLLMCombiner):
    def __init__(self, *args, dname='conll', **kwargs):
        self.data_name = dname
        BaseLLMCombiner.__init__(self, *args, **kwargs)

    def _make_all_changes(self, source_sentences, pred_texts):
        preds = list(zip(*pred_texts)) # outer iter over samples not models
        prompts = self._prep_prompts(source_sentences, preds)
        model_preds = self.comb_model.predict_all(prompts)

        # extract relevant part of output
        predictions = []
        for mpred, ps, s in zip(model_preds, preds, source_sentences):
            data_id = s.split(' ')[0]
            result = re.search('<output>(.*)</output>', mpred)
            if result is None:
                print("Failed format")
                result = ps[0] # select first prediction if failed output
            else:
                result = result.group(1)
                result = data_id + ' ' + normalize_text(result, data_name=self.data_name)
            predictions.append(result)
        return predictions

    def _prep_prompts(self, source_sentences, preds):
        prompts = [self._prompt(s, ps) for s,ps in zip(source_sentences, preds)]
        return prompts
    
    def _prompt(self, source, preds, remove_ids_in_prompt=True):
        if remove_ids_in_prompt:
            source = ' '.join(source.split(' ')[1:])

        if self.template == 0:
            # zero-shot
            out = (
                "You have to help with grammatical error correction."
                "You will be given the grammatically incorrect input sentence and three potential options for the grammatically corrected output sentence.\n"
                "Consider the different options and then combine them to give the correct grammatically corrected output sentence in the tags <output>corrected sentence</output>.\n"
                f"<input>{source}</input>\n"
            )
            for i, pred in enumerate(preds):
                if remove_ids_in_prompt:
                    pred = ' '.join(pred.split(' ')[1:])
                out += f"<option{i+1}>{pred}</option{i+1}>\n"
            out += 'Make sure to return the combined grammatically corrected ouput sentence in the tags <output>corrected sentence</output>.\n' 
        return out

