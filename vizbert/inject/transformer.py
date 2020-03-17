from scipy.linalg import orth
from transformers import GPT2Model
import torch
import torch.nn as nn

from vizbert.utils import orth_compl, sample_subspace_noise
from .base import InjectionHook, ForwardWrapper


class ProbeSubspaceNoiseModule(nn.Module):

    def __init__(self, probe_weight: torch.Tensor, use_complement=False, a=0.1):
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


class Gpt2NoiseInjectionHook(InjectionHook):

    def __init__(self, probe_noise: ProbeSubspaceNoiseModule, layer_idx: int, last_only=True):
        self.probe_noise = probe_noise
        self.last_only = last_only
        self.layer_idx = layer_idx

    def _routine(self, outputs):
        hid = outputs[0][-1]
        bsz = hid.size(0)
        hlen = hid.size(2)
        x1 = hid.size(1)
        x3 = hid.size(3)
        hid = hid.permute(0, 2, 1, 3).contiguous().view(bsz, hlen, -1)
        hid = self.probe_noise(hid)
        hid = hid.view(bsz, hlen, x1, x3).permute(0, 2, 1, 3).contiguous()
        outputs[0][-1] = hid
        return outputs

    def do_inject(self, model: GPT2Model):
        model.h[self.layer_idx] = ForwardWrapper(model.h[self.layer_idx], self._routine)

    def do_eject(self, model: GPT2Model):
        model.h[self.layer_idx] = model.h[self.layer_idx].module

    @property
    def name(self):
        return 'gpt2-probe-noise'
