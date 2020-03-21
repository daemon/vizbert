from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

from vizbert.utils import quick_nlp


__all__ = ['is_subword', 'SequentialLanguageModel']


def is_subword(token, scheme='gpt2'):
    if 'gpt2' in scheme:
        return not token.startswith('Ġ')
    elif 'bert' in scheme:
        return token.startswith('##')


@dataclass
class SequentialLanguageModel(object):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def condition_text(self, text: str) -> Any:
        """Conditions the model on text.
        :param text: the text to condition on.
        :return: an object representing the state of the language model.
        """
        raise NotImplementedError

    def compute_next_logits(self, state: Any) -> torch.Tensor:
        raise NotImplementedError

    def filtered_next_logits(self,
                             state: Any,
                             upos_filter=None,
                             singular_only=False,
                             plural_only=False,
                             feats_filter=None,
                             past_tokens=None,
                             whole_words_only=True):
        next_logits = self.compute_next_logits(state)
        mask = [1] * len(next_logits)
        if feats_filter is None:
            feats_filter = {}
        if singular_only:
            feats_filter['Number'] = 'Sing'
        if plural_only:
            feats_filter['Number'] = 'Plur'
        for idx, logit in enumerate(next_logits):
            token = self.tokenizer.convert_ids_to_tokens([idx])
            subword = is_subword(token, self.model.base_model_prefix)
            if whole_words_only and subword:
                mask[idx] = 0
            if upos_filter is not None or feats_filter:
                if subword:
                    mask[idx] = 0
                    continue
            else:
                continue
            nlp = quick_nlp(name='pos', tokenize_pretokenized=True)
            sent = nlp(past_tokens + [token])[0]
            if upos_filter and sent[-1].upos != upos_filter:
                mask[idx] = 0
                continue
            if any(sent[-1].feats.get(k) != v for k, v in feats_filter):
                mask[idx] = 0
                continue
        return next_logits, mask
