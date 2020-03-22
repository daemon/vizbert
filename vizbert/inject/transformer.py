from scipy.linalg import orth
from transformers import GPT2Model, BertModel
import torch
import torch.nn as nn

from vizbert.utils import orth_compl, sample_subspace_noise, batch_gs
from .base import InjectionHook, ForwardWrapper


__all__ = ['ProbeSubspaceNoiseModule',
           'Gpt2HiddenLayerInjectionHook',
           'ProbeDirectionRemovalModule',
           'BertHiddenLayerInjectionHook',
           'ProbeReportingModule']


class ProbeSubspaceNoiseModule(nn.Module):

    def __init__(self, probe_weight: torch.Tensor, use_complement=True, a=50):
        super().__init__()
        self.use_complement = use_complement
        self.a = a
        if use_complement:
            self.Q = orth_compl(probe_weight.cpu().numpy().T)[1]
        else:
            self.Q = orth(probe_weight.cpu().numpy().T)

    def forward(self, hidden_states: torch.Tensor):
        noise = sample_subspace_noise(self.Q, -self.a, self.a, hidden_states.size(1))
        return torch.from_numpy(noise).to(hidden_states.device) + hidden_states


class ProbeDirectionRemovalModule(nn.Module):

    def __init__(self,
                 probe_weight: torch.Tensor,
                 use_complement=False,
                 normalize=False,
                 strength=1):
        super().__init__()
        self.use_complement = use_complement
        if use_complement:
            Q = orth_compl(probe_weight.cpu().numpy().T)[1]
        else:
            Q = orth(probe_weight.cpu().numpy().T)
        Q = torch.from_numpy(Q)
        self.register_buffer('Q', Q)
        self.normalize = normalize
        self.strength = strength
        self.passthrough = False

    def forward(self, hidden_states: torch.Tensor):
        if self.passthrough:
            return hidden_states
        if self.normalize:
            old_norms = hidden_states.norm(dim=2).unsqueeze(-1)
        for q in self.Q.split(1, 1):
            q = q.contiguous().view(-1)
            hidden_states = batch_gs(q, hidden_states, strength=self.strength)
        if self.normalize:
            norms = hidden_states.norm(dim=2).unsqueeze(-1)
            hidden_states = (hidden_states / norms) * old_norms
        return hidden_states


class ProbeReportingModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.buffer = None

    def forward(self, hidden_states):
        self.buffer = hidden_states
        return hidden_states


class BertHiddenLayerInjectionHook(InjectionHook):

    def __init__(self, module: nn.Module, layer_idx: int):
        self.module = module
        self.layer_idx = layer_idx

    def _routine(self, outputs):
        hid = self.module(outputs[0])
        outputs = list(outputs)
        outputs[0] = hid
        return tuple(outputs)

    def do_inject(self, model: BertModel):
        model.encoder.layer[self.layer_idx] = ForwardWrapper(model.encoder.layer[self.layer_idx], self._routine)

    def do_eject(self, model: BertModel):
        model.encoder.layer[self.layer_idx] = model.encoder.layer[self.layer_idx].module

    @property
    def name(self):
        return f'bert-layer-inject-{self.layer_idx}'


class Gpt2HiddenLayerInjectionHook(InjectionHook):

    def __init__(self, module: nn.Module, layer_idx: int):
        self.module = module
        self.layer_idx = layer_idx

    def _routine(self, outputs):
        hid = self.module(outputs[0])
        outputs[0] = hid
        return outputs

    def do_inject(self, model: GPT2Model):
        model.h[self.layer_idx] = ForwardWrapper(model.h[self.layer_idx], self._routine)

    def do_eject(self, model: GPT2Model):
        model.h[self.layer_idx] = model.h[self.layer_idx].module

    @property
    def name(self):
        return f'gpt2-layer-inject-{self.layer_idx}'
