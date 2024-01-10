from tqdm import tqdm
import errant

class MBRcombiner:
    annotator = errant.load('en')

    def __init__(self, source_sentences, pred_texts):
        self.combined_texts = self._make_all_changes(source_sentences, pred_texts)

    @classmethod
    def _make_all_changes(cls, source_sentences, pred_texts):
        selected_samples = []
        # for n, samples in tqdm(enumerate(zip(*pred_texts)), total=len(source_sentences)):
        for n, samples in enumerate(zip(*pred_texts)):
            edits = [cls.return_edits(source_sentences[n], s) for s in samples]
            best = [None, -1] # [model index, score] 
            for i in range(len(edits)):
                total = 0
                for j in range(len(edits)):
                    if i == j:
                        continue
                    score = cls.edit_f05(edits[j], edits[i])
                    total += score
                if total > best[1]:
                    best = [i, total]
            selected_samples.append(samples[best[0]])
        return selected_samples

    @classmethod
    def return_edits(cls, input, prediction):
        '''
        Get edits
        '''
        input = cls.annotator.parse(input)
        prediction = cls.annotator.parse(prediction)
        alignment = cls.annotator.align(input, prediction)
        edits = cls.annotator.merge(alignment)
        for e in edits:
            e = cls.annotator.classify(e)
        return edits

    @staticmethod
    def edit_agreement(edits1, edits2):
        '''
            Number of matching edits 
        '''
        edits1_str = [e.o_str+' -> '+e.c_str for e in edits1]
        edits2_str = [e.o_str+' -> '+e.c_str for e in edits2]

        matched = 0
        for e1_str in edits1_str:
            if e1_str in edits2_str:
                matched += 1
        return matched

    @staticmethod
    def edit_jaccard(edits1, edits2):
        list1 = [e.o_str+' -> '+e.c_str for e in edits1]
        list2 = [e.o_str+' -> '+e.c_str for e in edits2]

        if len(list1) == 0 and len(list2) == 0:
            return 1

        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection

        return float(intersection) / union

    @staticmethod
    def edit_rec(edits1, edits2):

        # rec estimate based reward
        # edits1: ref 
        # edits2: hyp
        list1 = [e.o_str+' -> '+e.c_str for e in edits1]
        list2 = [e.o_str+' -> '+e.c_str for e in edits2]

        if len(list1) == 0:
            return 1

        intersection = len(list(set(list1).intersection(list2)))
        return float(intersection) / len(list1)

    @staticmethod
    def edit_prec(edits1, edits2):

        # prec estimate based reward
        # edits1: ref 
        # edits2: hyp
        list1 = [e.o_str+' -> '+e.c_str for e in edits1]
        list2 = [e.o_str+' -> '+e.c_str for e in edits2]

        if len(list2) == 0:
            return 1

        intersection = len(list(set(list1).intersection(list2)))
        return float(intersection) / len(list2)

    @staticmethod
    def edit_f05(edits1, edits2):
        # f05 estimate based reward
        # edits1: ref 
        # edits2: hyp
        k = 0.5
        list1 = [e.o_str+' -> '+e.c_str for e in edits1]
        list2 = [e.o_str+' -> '+e.c_str for e in edits2]
        if len(list1 + list2) == 0:
            return 1
        intersection = len(list(set(list1).intersection(list2)))
        return ((1+(k**2))*intersection)/((k*len(list1)) + len(list2))