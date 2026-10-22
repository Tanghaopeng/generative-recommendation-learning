"""BARGE-inspired item-context attention for historical SID groups in Qwen2.

The assistant boundary is mandatory: teacher-forced answers are NEVER pooled.
This is a local decoder-only adaptation, not the paper authors' implementation.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Model, Qwen2PreTrainedModel

ICA_PROTOCOL = 'historical_sid_ica_v1: pool semantic 0-2; fuse historical 0-3; answer excluded'


def tokenizer_binding(tokenizer, width):
    namespaces = [[tokenizer.convert_tokens_to_ids(f'<sid_{layer}_{k}>')
                   for k in range(width)] for layer in range(4)]
    flat = sum(namespaces, [])
    if len(set(flat)) != 4 * width or any(i is None or i == tokenizer.unk_token_id for i in flat):
        raise ValueError('Existing SFT tokenizer must contain four complete atomic SID namespaces')
    # Use chat_template itself, including the assistant role and its final newline.
    messages = [dict(role='user', content='binding check')]
    closed = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    opened = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if opened[:len(closed)] != closed or not opened[len(closed):]:
        raise ValueError('Cannot identify the assistant generation delimiter')
    return dict(sid_token_ids=namespaces, assistant_delimiter=opened[len(closed):])


class QwenICAConfig(Qwen2Config):
    model_type = 'qwen2_ica'

    def __init__(self, ica=None, **kwargs):
        super().__init__(**kwargs)
        self.ica = copy.deepcopy(ica) if ica is not None else None


class ItemContextAttention(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.enabled = config.get('enabled', True)
        dim, heads, bottleneck = (config[k] for k in ('attention_dim', 'heads', 'bottleneck'))
        if dim < 1 or heads < 1 or dim % heads or bottleneck < 1:
            raise ValueError('ICA requires positive dimensions and attention_dim divisible by heads')
        # A low-width attention projection keeps the added module small.
        self.project = nn.Linear(hidden_size, dim, bias=False)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.restore = nn.Linear(dim, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp_in = nn.Linear(hidden_size, bottleneck)
        self.mlp_out = nn.Linear(bottleneck, hidden_size)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)

    def reset_residual(self):
        # Identity at insertion. Only the output projection learns on the first step;
        # the rest of the ICA branch receives gradients after it becomes nonzero.
        nn.init.zeros_(self.mlp_out.weight)
        nn.init.zeros_(self.mlp_out.bias)

    def forward(self, grouped):
        # grouped: [number_of_history_items, 4, Qwen_hidden]. Fourth is collision ID.
        keys = self.project(grouped[:, :3])
        query = self.query.expand(len(grouped), -1, -1)
        pooled = self.attention(query, keys, keys, need_weights=False)[0][:, 0]
        summary = self.mlp_out(torch.nn.functional.gelu(self.mlp_in(self.norm(self.restore(pooled)))))
        repeated = summary[:, None, :].expand(-1, 4, -1)
        gate = torch.sigmoid(self.gate(torch.cat([grouped, repeated], dim=-1)))
        return gate * repeated


class QwenICAModel(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        if not config.ica:
            raise ValueError('Missing ICA architecture/tokenizer binding')
        self.ica = ItemContextAttention(config.hidden_size, config.ica)
        namespace = torch.full((config.vocab_size,), -1, dtype=torch.long)
        for layer, ids in enumerate(config.ica['sid_token_ids']):
            namespace[ids] = layer
        self.register_buffer('sid_namespace', namespace, persistent=False)
        self.register_buffer('assistant_delimiter', torch.tensor(config.ica['assistant_delimiter']), persistent=False)

    def history_groups(self, input_ids, attention_mask=None):
        """Return [item, (batch_row, start_column)]. GPU-side parsing, no target data.

        Require exactly one assistant delimiter in a complete prompt. A cached
        one-token decoding step has no complete four-token item and returns empty.
        """
        length = input_ids.shape[1]
        if length < max(4, self.assistant_delimiter.numel()):
            return torch.empty((0, 2), dtype=torch.long, device=input_ids.device)
        marker = (input_ids.unfold(1, len(self.assistant_delimiter), 1)
                  == self.assistant_delimiter).all(-1)
        if attention_mask is not None:
            marker &= attention_mask.unfold(1, len(self.assistant_delimiter), 1).bool().all(-1)
        counts = marker.sum(1)
        if not torch.all(counts == 1):
            raise ValueError('ICA accepts one next-item assistant prompt per row; no packed/multi-turn samples')
        boundary = marker.long().argmax(1)
        slots = self.sid_namespace[input_ids].unfold(1, 4, 1)
        match = (slots == torch.arange(4, device=input_ids.device)).all(-1)
        match &= torch.arange(length - 3, device=input_ids.device)[None, :] + 4 <= boundary[:, None]
        if attention_mask is not None:
            match &= attention_mask.unfold(1, 4, 1).bool().all(-1)
        return match.nonzero(as_tuple=False)

    def contextual_embeddings(self, input_ids, attention_mask=None):
        embeddings = self.embed_tokens(input_ids)
        if not self.ica.enabled:
            return embeddings
        groups = self.history_groups(input_ids, attention_mask)
        if not len(groups):
            return embeddings
        rows, starts = groups.unbind(1)
        columns = starts[:, None] + torch.arange(4, device=input_ids.device)
        grouped = embeddings[rows[:, None], columns]
        delta = self.ica(grouped).to(embeddings.dtype)
        # Groups do not overlap. CopySlices preserves gradients into Qwen and ICA.
        output = embeddings.clone()
        output[rows[:, None], columns] = grouped + delta
        return output

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, cache_position=None, **kwargs):
        if inputs_embeds is not None:
            raise ValueError('ICA requires input_ids to locate historical groups; external inputs_embeds disabled')
        if input_ids is None:
            raise ValueError('ICA requires token IDs')
        # Cached new tokens are causally processed without pooling generated SIDs.
        cached = past_key_values is not None and past_key_values.get_seq_length() > 0
        embeddings = self.embed_tokens(input_ids) if cached else self.contextual_embeddings(input_ids, attention_mask)
        return super().forward(input_ids=None, inputs_embeds=embeddings,
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, cache_position=cache_position, **kwargs)


class QwenICAForCausalLM(Qwen2ForCausalLM):
    config_class = QwenICAConfig

    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = QwenICAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.model.ica.reset_residual()

    @classmethod
    def from_sft(cls, directory, tokenizer, width, settings, **kwargs):
        base = Qwen2Config.from_pretrained(directory, local_files_only=True)
        if base.model_type != 'qwen2' or getattr(base, 'ica', None):
            raise ValueError('Initialize ICA once from an ordinary trained Qwen2 SID SFT checkpoint')
        if base.vocab_size != len(tokenizer):
            raise ValueError('SFT vocabulary mismatch; resizing or Q0 initialization is forbidden here')
        config = base.to_dict()
        config.pop('model_type', None)
        config = QwenICAConfig(**config, ica=dict(settings, **tokenizer_binding(tokenizer, width)))
        model = cls.from_pretrained(directory, config=config, local_files_only=True, **kwargs)
        # HF low-memory loading initializes missing ordinary Linear layers but
        # does not know MHA's raw in_proj_weight/query parameters. Initialize the
        # entire NEW module explicitly, never touch already loaded SFT parameters.
        reference = model.get_input_embeddings().weight
        model.model.ica = ItemContextAttention(config.hidden_size, config.ica).to(
            device=reference.device, dtype=reference.dtype)
        model.model.ica.reset_residual()
        return model


def load_ica_tokenizer(directory):
    return AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                         fix_mistral_regex=False)


# Registration is explicit: loading a saved ICA package without importing this
# module fails clearly rather than silently dropping the learned ICA weights.
AutoConfig.register('qwen2_ica', QwenICAConfig)
AutoModelForCausalLM.register(QwenICAConfig, QwenICAForCausalLM)
