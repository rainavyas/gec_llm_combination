'''
Using LLM to combine model outputs
'''
from abc import ABC, abstractmethod
from .select_llm_base_model import get_model

class BaseLLMCombiner(ABC):
    def __init__(self, source_sentences, pred_texts, comb_model_name='mistral-7b', gpu_id=0):
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
        prompts = self._prep_prompts(source_sentences, pred_texts)
        selections = self.comb_model.predict_all(prompts)
        return selections # to change

    def _prep_prompts(self, source_sentences, pred_texts):
        preds = list(zip(*pred_texts))
        prompts = [self._prompt(s, ps) for s,ps in zip(source_sentences, preds)]
        return prompts
    
    def _prompt(self, source, preds):
        out = (
            "Select an output sentence option that is the best for grammatical error correction "
            "of an input sentence. Select only one output sentence option from {1,2,3} and return only the option number. Do not give an explanation.\n"
            f"Input: {source}\n"
        )
        for i, pred in enumerate(preds):
            out += f"<option {i+1}: {pred}\n"
        out += f"\n\n"
        out+='The best option number is: '
        return out

