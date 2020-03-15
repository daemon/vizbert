from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from tqdm import tqdm
from transformers import PreTrainedModel
import torch

from vizbert.data import TextBatch


class InputFeeder(object):

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def feed_input(self, model, inp):
        raise NotImplementedError


class OutputTransform(object):

    def __call__(self, model, output):
        raise NotImplementedError


class OutputExtractor(object):

    def __call__(self, output):
        raise NotImplementedError


class OutputSerializer(object):

    def __call__(self, output_dict, inp):
        raise NotImplementedError

    def finalize(self):
        pass


class TransformerInputFeeder(InputFeeder):

    def __init__(self, dataloader, device='cpu'):
        self.dataloader = dataloader
        self.device = torch.device(device)

    def __len__(self):
        return len(self.dataloader)

    def __next__(self) -> TextBatch:
        return next(self._iter)

    def feed_input(self, model, inp: TextBatch):
        model.eval()
        with torch.no_grad():
            other_data = dict()
            if model.base_model_prefix == 'bert':
                other_data = dict(token_type_ids=torch.zeros_like(inp.token_ids).to(self.device))
            outputs = model(inp.token_ids.to(self.device),
                            attention_mask=inp.attention_mask.to(self.device),
                            **other_data)
        return outputs

    def __iter__(self):
        self._iter = iter(self.dataloader)
        return self


class IdentityOutputExtractor(OutputExtractor):

    def __call__(self, output):
        return dict(output=output)


class BufferedFileOutputSerializer(OutputSerializer):

    def __init__(self, filename):
        self.filename = filename
        self.buffered_dict = defaultdict(list)

    def __call__(self, output_dict, inp):
        for k, v in output_dict.items():
            self.buffered_dict[k].append(v)

    def finalize(self):
        torch.save(self.buffered_dict, self.filename)


@dataclass
class ModelStateExtractor(object):
    model: PreTrainedModel
    input_feeder: InputFeeder
    output_serializer: OutputSerializer
    output_transform: OutputTransform = None
    output_extractors: Sequence[OutputExtractor] = field(default_factory=lambda: [IdentityOutputExtractor()])

    def extract(self, use_tqdm=True):
        for inp in tqdm(iter(self.input_feeder), total=len(self.input_feeder), disable=not use_tqdm):
            output = self.input_feeder.feed_input(self.model, inp)
            output_dict = {}
            for extractor in self.output_extractors:
                output_dict.update(extractor(output))
            if self.output_transform is not None:
                self.output_transform(output_dict)
            self.output_serializer(output_dict, inp)
        self.output_serializer.finalize()
