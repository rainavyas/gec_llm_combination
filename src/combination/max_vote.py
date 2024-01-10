import numpy as np
from tqdm import tqdm
import spacy
from difflib import SequenceMatcher
from collections import Counter

class Maxvotecombiner:
    '''
        Code inspired by: https://github.com/MaksTarnavskyi/gector-large/blob/master/ensemble.py
    '''
    def __init__(self, source_sentences, pred_texts, min_count=2):
        self.combined_texts = self._make_all_changes(source_sentences, pred_texts, min_count=min_count)
    
    @classmethod
    def _make_all_changes(cls, source_sentences, pred_texts, min_count=2):
        nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'textcat', 'custom'])
        pred_texts = np.array(pred_texts)
        sent_after_merge = []
        # for i in tqdm(range(len(source_sentences))):
        for i in range(len(source_sentences)):
            source_sentence = source_sentences[i]
            target_sentences = pred_texts[:, i]
            new_sentence = cls._make_changes(nlp, source_sentence, target_sentences=target_sentences, min_count=min_count, debug=False)
            sent_after_merge.append(new_sentence)
        return sent_after_merge
    
    @classmethod
    def _make_changes(cls, nlp, source_sentence, target_sentences=[], min_count=2, debug=False):
        source_tokens = cls.get_tokens(nlp(str(source_sentence)))

        target_docs_tokens = [cls.get_tokens(nlp(str(sent))) for sent in target_sentences]
        all_actions = []

        for i in range(len(target_sentences)):

            target_tokens = target_docs_tokens[i]

            matcher = SequenceMatcher(None, source_tokens, target_tokens)

            raw_diffs = list(matcher.get_opcodes())

            for diff in raw_diffs:
                if diff[0] == 'replace':
                    # "source_start_token", "source_end_token", "target_part"
                    all_actions.append(
                        ('replace', diff[1], diff[2], "".join(target_tokens[diff[3]: diff[4]]))
                    )
                if diff[0] == 'delete':
                    # "source_start_token", "source_end_token"
                    all_actions.append(
                        ('delete', diff[1], diff[2])
                    )
                if diff[0] == 'insert':
                    # "source_start_token", "target_part"
                    all_actions.append(
                        ('insert', diff[1], "".join(target_tokens[diff[3]: diff[4]]))
                    )

        good_actions = [k for k, v in Counter(all_actions).items() if v >= min_count]
        good_actions.sort(key=lambda x: x[1])  # sort by second field - start token

        if debug:
            print("All actions", all_actions)
            print("Good actions", good_actions)

        if len(good_actions) > 0:

            final_text = ""
            current_start = 0
            previous_end = 0

            for action in good_actions:
                current_start = action[1]
                final_text += "".join(source_tokens[previous_end: current_start])
                if action[0] == 'replace':
                    final_text += action[3]
                    previous_end = action[2]
                if action[0] == 'delete':
                    previous_end = action[2]
                if action[0] == 'insert':
                    final_text += action[2]
                    previous_end = action[1]

            final_text += "".join(source_tokens[previous_end:])
            return final_text

        else:
            return ''.join(source_tokens)

    @staticmethod
    def get_tokens(doc):
        all_tokens = []
        for token in doc:
            all_tokens.append(token.text)
            if len(token.whitespace_):
                all_tokens.append(token.whitespace_)
        return all_tokens