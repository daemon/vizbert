from dataclasses import dataclass

from transformers import GPT2LMHeadModel, GPT2Tokenizer


__all__ = ['EnhancedGpt2Generator']


@dataclass
class EnhancedGpt2Generator(object):
    model: GPT2LMHeadModel
    tokenizer: GPT2Tokenizer
    whole_words_only: bool = False

    def generate(self, condition_text: str):
        self.tokenizer.tokenize(condition_text)
