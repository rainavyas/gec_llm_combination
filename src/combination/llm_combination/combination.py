'''
Using LLM to combine model outputs
'''
from abc import ABC, abstractmethod
from .select_llm_base_model import get_model

class BaseLLMCombiner(ABC):
    def __init__(self, source_sentences, pred_texts, comb_model_name='mistral-7b'):
        self.comb_model = get_model(comb_model_name)
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

    def _prep_prompts(self, source_sentences, pred_texts):
        preds = list(zip(*pred_texts))
        prompts = [self._prompt(s, ps) for s,ps in zip(source_sentences, preds)]
        return prompts
    
    def _prompt(self, source, preds):
        out = (
            "You are to help with grammatical error correction."
            "You will be given a source grammatically incorrect sentence (src) and possible"
            "grammatically corrected output sentence options (option), "
            "select a single output sentence option that best perform grammatical error correction of the source sentence."
            "Make sure you return only the selected output sentence and nothing else.\n"
            f"src: {source}\n"
        )
        for i, pred in enumerate(preds):
            out += f"output{i+1}: {pred}\n"
        out += f"\n"
        return out

