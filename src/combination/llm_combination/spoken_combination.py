from src.tools.tools import spoken_gec_normalize_text
from .select_llm_base_model import get_model

import re


class SpokenLLMCombiner():
    def __init__(self, pred_texts, comb_model_name='mistral-7b', gpu_id=0, template=2):
        self.template = template
        self.comb_model = get_model(comb_model_name, gpu_id=gpu_id)
        self.combined_texts = self._make_all_changes(pred_texts)

    def _make_all_changes(self, pred_texts):
        '''
        pred_texts order: disfluent, fluent, gec_pred
        '''
        preds = list(zip(*pred_texts)) # outer iter over samples not models
        prompts = self._prep_prompts(preds)
        model_preds = self.comb_model.predict_all(prompts)

        # extract relevant part of output
        predictions = []
        for mpred, ps in zip(model_preds, preds):
            data_id = ps[0].split(' ')[0]
            result = re.search('<output>(.*)</output>', mpred)
            if result is None:
                print("Failed format")
                result = ps[0] # select first prediction if failed output
            else:
                result = result.group(1)
                result = data_id + ' ' + spoken_gec_normalize_text(result)
            predictions.append(result)
        return predictions

    def _prep_prompts(self, preds):
        prompts = [self._prompt(ps) for ps in preds]
        return prompts

    def _prompt(self, preds, remove_ids_in_prompt=True):

        if remove_ids_in_prompt:
            new_preds = []
            for pred in preds:
                new_preds.append(' '.join(pred.split(' ')[1:]))
        else:
            new_preds = preds

        if self.template == 0:
            # zero-shot
            out = (
                "You have to help with spoken grammatical error correction."
                "You will be given three text views of the spoken audio: disfluent transcription <dsf>, fluent transcription <flt> and the grammatical error correction prediction <gec>.\n"
                "Consider the different views and then combine them to give the grammatically corrected output sentence in the tags <output>corrected sentence</output>.\n\n"
                f"<dsf>{new_preds[0]}</dsf>\n"
                f"<flt>{new_preds[1]}</flt>\n"
                f"<gec>{new_preds[2]}</gec>\n\n"
                "Make sure to return the combined grammatically corrected ouput sentence in the tags <output>corrected sentence</output>.\n"
                )
        return out

