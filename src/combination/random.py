import random

class RandCombiner:
    '''
       Randomly select model for output each turn
    '''
    def __init__(self, pred_texts):
        self.combined_texts = self._make_all_changes(pred_texts)
    
    @staticmethod
    def _make_all_changes(pred_texts):
        preds = list(zip(*pred_texts))
        outputs = []
        for ps in preds:
            ind = random.randint(0,2)
            outputs.append(ps[ind])
        return outputs
