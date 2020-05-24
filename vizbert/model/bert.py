from itertools import chain
import math

from transformers.modeling_bert import BertLayerNorm, ACT2FN, BertOutput
import torch
from torch import nn

from .probe import LowRankProjectionTransform, LowRankLinear


__all__ = ['LowRankBertOutput', 'LowRankBertSelfOutput', 'LowRankBertSelfAttention', 'LowRankBertIntermediate',
           'LLowRankBertIntermediate', 'LLowRankBertOutput', 'LLowRankBertSelfAttention', 'LLowRankBertSelfOutput']


class LLowRankBertOutput(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.low_dense = LowRankLinear(config.intermediate_size, rank, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def low_rank_parameters(self):
        return self.low_dense.parameters()

    @torch.no_grad()
    def init_pretrained(self, module):
        self.low_dense.init_pretrained(module.dense)
        self.LayerNorm.weight.set_(module.LayerNorm.weight)
        self.LayerNorm.bias.set_(module.LayerNorm.bias)
        return self

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.low_dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LLowRankBertIntermediate(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.low_dense = LowRankLinear(config.hidden_size, rank, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def low_rank_parameters(self):
        return self.low_dense.parameters()

    @torch.no_grad()
    def init_pretrained(self, module):
        self.low_dense.init_pretrained(module.dense)
        return self

    def forward(self, hidden_states):
        hidden_states = self.low_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LLowRankBertSelfOutput(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.low_dense = LowRankLinear(config.hidden_size, rank, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def low_rank_parameters(self):
        return self.low_dense.parameters()

    @torch.no_grad()
    def init_pretrained(self, module):
        self.low_dense.init_pretrained(module.dense)
        self.LayerNorm.weight.set_(module.LayerNorm.weight)
        self.LayerNorm.bias.set_(module.LayerNorm.bias)
        return self

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.low_dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LLowRankBertSelfAttention(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.low_q = LowRankLinear(config.hidden_size, rank, self.all_head_size)
        self.low_k = LowRankLinear(config.hidden_size, rank, self.all_head_size)
        self.low_v = LowRankLinear(config.hidden_size, rank, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def low_rank_parameters(self):
        return chain(self.low_k.parameters(), self.low_q.parameters(), self.low_v.parameters())

    @torch.no_grad()
    def init_pretrained(self, module):
        self.low_q.init_pretrained(module.query)
        self.low_k.init_pretrained(module.key)
        self.low_v.init_pretrained(module.value)
        return self

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.low_q(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.low_k(encoder_hidden_states)
            mixed_value_layer = self.low_v(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.low_k(hidden_states)
            mixed_value_layer = self.low_v(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class LowRankBertOutput(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.low = LowRankProjectionTransform(config.hidden_size, rank)
        self.low_dense1 = nn.Linear(config.intermediate_size, rank, bias=False)
        self.low_dense2 = nn.Linear(rank, config.intermediate_size)
        self.applied = False

    def low_rank_parameters(self):
        return self.low.parameters()

    def apply_low_rank(self):
        self.applied = True
        with torch.no_grad():
            self.low_dense1.weight.set_(self.low.orth_probe)
            self.low_dense2.weight.set_(self.dense.weight.matmul(self.low_dense1.weight))
            self.low_dense2.bias.set_(self.dense.bias)

    @torch.no_grad()
    def init_pretrained(self, module):
        self.dense.weight.set_(module.dense.weight)
        self.dense.bias.set_(module.dense.bias)
        self.LayerNorm.weight.set_(module.LayerNorm.weight)
        self.LayerNorm.bias.set_(module.LayerNorm.bias)
        return self

    def forward(self, hidden_states, input_tensor):
        if self.applied:
            hidden_states = self.low_dense2(self.low_dense1(hidden_states))
        else:
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return self.low(hidden_states)


class LowRankBertIntermediate(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.low = LowRankProjectionTransform(config.hidden_size, rank)
        if rank > 0:
            self.low_dense1 = nn.Linear(config.hidden_size, rank, bias=False)
            self.low_dense2 = nn.Linear(rank, config.hidden_size)
        self.int_size = config.intermediate_size
        self.rank = rank
        self.applied = False

    def low_rank_parameters(self):
        if self.rank == 0:
            return [self.dense.bias]
        return self.low.parameters()

    @torch.no_grad()
    def init_pretrained(self, module):
        self.dense.weight.set_(module.dense.weight)
        self.dense.bias.set_(module.dense.bias)
        return self

    def apply_low_rank(self):
        self.applied = True
        with torch.no_grad():
            self.low_dense1.weight.set_(self.low.orth_probe)
            self.low_dense2.weight.set_(self.dense.weight.matmul(self.low_dense1.weight))
            self.low_dense2.bias.set_(self.dense.bias)

    def forward(self, hidden_states):
        if self.applied:
            hidden_states = self.low_dense2(self.low_dense1(hidden_states))
        else:
            if self.rank > 0:
                hidden_states = self.dense(self.low(hidden_states))
            else:
                hidden_states = torch.zeros(hidden_states.size(0), hidden_states.size(1), self.int_size).to(hidden_states.device) + self.dense.bias
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LowRankBertSelfOutput(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.low = LowRankProjectionTransform(config.hidden_size, rank)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.low_dense1 = nn.Linear(config.hidden_size, rank, bias=False)
        self.low_dense2 = nn.Linear(rank, config.hidden_size)
        self.applied = False

    def low_rank_parameters(self):
        return self.low.parameters()

    @torch.no_grad()
    def init_pretrained(self, module):
        self.dense.weight.set_(module.dense.weight)
        self.dense.bias.set_(module.dense.bias)
        self.LayerNorm.weight.set_(module.LayerNorm.weight)
        self.LayerNorm.bias.set_(module.LayerNorm.bias)
        return self

    def apply_low_rank(self):
        self.applied = True
        with torch.no_grad():
            self.low_dense1.weight.set_(self.low.orth_probe)
            self.low_dense2.weight.set_(self.dense.weight.matmul(self.low_dense1.weight))
            self.low_dense2.bias.set_(self.dense.bias)

    def forward(self, hidden_states, input_tensor):
        if self.applied:
            hidden_states = self.low_dense2(self.low_dense1(hidden_states))
        else:
            # hidden_states = self.low(hidden_states)
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return self.low(hidden_states)


class LowRankBertSelfAttention(nn.Module):
    def __init__(self, config, rank):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.low_q = LowRankProjectionTransform(config.hidden_size, rank)
        self.low_k = LowRankProjectionTransform(config.hidden_size, rank)
        self.low_v = LowRankProjectionTransform(config.hidden_size, rank)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.low_hid = nn.Linear(config.hidden_size, rank, bias=False)
        self.low_query = nn.Linear(rank, config.hidden_size)
        self.low_key = nn.Linear(rank, config.hidden_size)
        self.low_value = nn.Linear(rank, config.hidden_size)
        self.applied = False

    def low_rank_parameters(self):
        return chain(self.low_k.parameters(), self.low_q.parameters(), self.low_v.parameters())

    @torch.no_grad()
    def init_pretrained(self, module):
        self.query.weight.set_(module.query.weight)
        self.key.weight.set_(module.key.weight)
        self.value.weight.set_(module.value.weight)
        self.query.bias.set_(module.query.bias)
        self.key.bias.set_(module.key.bias)
        self.value.bias.set_(module.value.bias)
        return self

    def apply_low_rank(self):
        self.applied = True
        with torch.no_grad():
            self.low_hid.weight.set_(self.low.orth_probe)
            self.low_query.weight.set_(self.query.weight.matmul(self.low_hid.weight))
            self.low_key.weight.set_(self.key.weight.matmul(self.low_hid.weight))
            self.low_value.weight.set_(self.value.weight.matmul(self.low_hid.weight))
            self.low_query.bias.set_(self.query.bias)
            self.low_key.bias.set_(self.key.bias)
            self.low_value.bias.set_(self.value.bias)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        if self.applied:
            low_hid = self.low_hid(hidden_states)
            mixed_query_layer = self.low_query(low_hid)
        else:
            mixed_query_layer = self.query(self.low_q(hidden_states))

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            if self.applied:
                mixed_key_layer = self.low_key(low_hid)
                mixed_value_layer = self.low_value(low_hid)
            else:
                mixed_key_layer = self.key(self.low_k(hidden_states))
                mixed_value_layer = self.value(self.low_v(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs
