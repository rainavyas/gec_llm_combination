from src.combination.mbr import MBRcombiner
from src.combination.max_vote import Maxvotecombiner
import errant

class BaseEnsModel:
    annotator = errant.load('en')

    def __init__(self, ind_models):
        self.ind_models = ind_models
    
    def _ind_model_preds(self, data):
        pred_texts = []
        for model in self.ind_models:
            pred_texts.append(model.predict(data))
        return pred_texts

    def predict(self, data, return_cnt=False):
        pred_texts = self._ind_model_preds(data)
        combined_texts = self._make_all_changes(data, pred_texts)

        if return_cnt:
            return combined_texts, self._count_edits(data, combined_texts)
        else:
            return combined_texts
    
    @classmethod
    def _count_edits(cls, source_sentences, pred_sentences):
        total_edits = 0
        for s,p in zip(source_sentences, pred_sentences):
            input = cls.annotator.parse(s)
            prediction = cls.annotator.parse(p)
            alignment = cls.annotator.align(input, prediction)
            edits = cls.annotator.merge(alignment)
            total_edits += len(edits)
        return total_edits

class MBREnsModel(BaseEnsModel, MBRcombiner):
    def __init__(self, ind_models):
        BaseEnsModel.__init__(self, ind_models)

class MaxvoteEnsModel(BaseEnsModel, Maxvotecombiner):
    def __init__(self, ind_models):
        BaseEnsModel.__init__(self, ind_models)
    
    
