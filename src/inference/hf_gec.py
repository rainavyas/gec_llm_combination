from src.combination.llm_combination.hf_model import HFModel
from src.tools.tools import spacy_normalize_text

from tqdm import tqdm
import re

class HFGECModel:
    def __init__(self, device, model_name="mistral-7b"):
        self.model = HFModel(device, model_name=model_name)
    
    def predict(self, data, has_id=True):
        '''
            has_id=True assumes the data is in the form:
                id1 sentence1
                id2 sentence2
                .
                .
                .
        '''
        predictions = []
        # need to remove ids if has_id and add back at the end
        ids = []
        for sent in tqdm(data):
            parts = sent.split(' ')
            curr_sent = ' '.join(parts[1:])
            ids.append(parts[0])
            prompt = self._prep_prompt(curr_sent)
            pred = self.model.predict(prompt)
            result = re.search('<output>(.*)</output>', pred)
            if result is None:
                print("Failed format")
                result = curr_sent
            else:
                result = result.group(1)
            
            # normalize to have space before punctuation
            result = spacy_normalize_text(result)
            predictions.append(result)
            # breakpoint()

        # Add back ids
        final_preds = [i+' '+p for i,p in zip(ids, predictions)]
        return final_preds

    def _prep_prompt(self, sentence):
        prompt = (
            "Perform grammatical error correction. " 
            "For the provided input sentence, give only the grammatically corrected output sentence in the tags <output>corrected sentence</output>.\n"
            f"<input>{sentence}</input>\n"
            "Make sure to return the grammatically corrected output sentence in the tags <output>corrected sentence</output>\n"
        )
        return prompt


