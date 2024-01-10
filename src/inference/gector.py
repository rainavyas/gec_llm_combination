from tqdm import tqdm

from gector.gec_model import GecBERTModel

class GectorModel:
    def __init__(self, args, transformer_model='roberta', model_path=None):
        if transformer_model == 'roberta':
            special_tokens_fix = 1
        else:
            special_tokens_fix = 0
        
        # # get model tweaking hyperparameters
        # if transformer_model == 'bert':
        #     confidence_bias = 0.1
        #     mep = 0.41
        # elif transformer_model == 'roberta':
        #     confidence_bias = 0.2
        #     mep = 0.50
        # elif transformer_model == 'xlnet':
        #     confidence_bias = 0.35
        #     mep = 0.66
        confidence_bias = 0
        mep = 0

        self.model = GecBERTModel(vocab_path=args.vocab_path,
                         model_paths=[model_path],
                         max_len=args.max_len, min_len=args.min_len,
                         iterations=args.iteration_count,
                         min_error_probability=mep,
                         lowercase_tokens=args.lowercase_tokens,
                         model_name=transformer_model,
                         special_tokens_fix=special_tokens_fix,
                         log=False,
                         confidence=confidence_bias,
                         del_confidence=args.additional_del_confidence,
                         is_ensemble=False,
                         weigths=args.weights)
        
        self.args = args
    
    def predict(self, data, return_cnt=False):
        predictions = []
        cnt_corrections = 0
        batch = []
        for sent in data:
            batch.append(sent.split())
            if len(batch) == self.args.batch_size:
                preds, cnt = self.model.handle_batch(batch)
                predictions.extend(preds)
                cnt_corrections += cnt
                batch = []
        if batch:
            preds, cnt = self.model.handle_batch(batch)
            predictions.extend(preds)
            cnt_corrections += cnt

        # print(f"Produced overall corrections: {cnt_corrections}")
        result_lines = [" ".join(x) for x in predictions]
        if return_cnt:
            return result_lines, cnt_corrections
        return result_lines