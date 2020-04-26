from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
import re

from .classification import SENTENCE_SEPARATOR


__all__ = ['ImdbSpuriousMatcher']
digits = ['1', 'one', 'two', '2', 'three', '3', '4', 'four', 'five', '5', '6', 'six', 'seven', '7', 'eight', '8',
          '9', 'nine', 'ten', '10']
digit_patt = '(' + '|'.join(digits) + ')'
slash_patt = re.compile(rf'((\d+(\.\d*)?|{digit_patt})\s*/\s*(10|ten)|\d+(\.\d*)?)', re.IGNORECASE)
out_of_patt = re.compile(rf'{digit_patt} out of (ten|10)', re.IGNORECASE)


@dataclass
class ImdbSpuriousMatcher(object):
    sentence_cutoff: int = 3
    use_slash_rule: bool = True
    use_out_of_rule: bool = True

    def match(self, text):
        sents = text.split(SENTENCE_SEPARATOR)[-self.sentence_cutoff:]
        match_map = defaultdict(list)
        for sent in sents:
            if self.use_out_of_rule:
                match_map['out_of'].extend(out_of_patt.finditer(sent))
            if self.use_slash_rule:
                match_map['slash'].extend(slash_patt.finditer(sent))
        return match_map

    def clean(self, text):
        splits = text.split(SENTENCE_SEPARATOR)
        prev_text = SENTENCE_SEPARATOR.join(splits[:-self.sentence_cutoff])
        text = f'{SENTENCE_SEPARATOR}{SENTENCE_SEPARATOR.join(splits[-self.sentence_cutoff:])}'
        if self.use_out_of_rule:
            text = out_of_patt.sub('', text)
        if self.use_slash_rule:
            text = slash_patt.sub('', text)
        return f'{prev_text}{text}'
