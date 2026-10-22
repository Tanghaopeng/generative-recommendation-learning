"""Joint next-item CE + HPR training: gradients update ICA, Qwen and HPR together.

No frozen context cache. HPR uses the last PROMPT state, never an answer state.
This is a local BARGE-inspired decoder-only adaptation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel

from qwen_ica import QwenICAConfig, QwenICAForCausalLM, QwenICAModel, ItemContextAttention, tokenizer_binding
from qwen_hpr import HPRConfig, HPRScorer, PrefixNegativeSampler, contrastive_loss

JOINT_PROTOCOL = 'ica_hpr_joint_v1: answer CE + semantic-prefix contrastive; all Qwen/ICA/HPR trainable'


class QwenICAHPRConfig(QwenICAConfig):
    model_type = 'qwen2_ica_hpr_joint'

    def __init__(self, joint=None, **kwargs):
        super().__init__(**kwargs)
        self.joint = dict(joint or {})


@dataclass
class JointOutput(CausalLMOutputWithPast):
    ce_loss: torch.Tensor | None = None
    hpr_loss: torch.Tensor | None = None
    prompt_context: torch.Tensor | None = None


class QwenICAHPRForCausalLM(QwenICAForCausalLM):
    config_class = QwenICAHPRConfig
    # The non-strict HF flag permits BF16 conversion. Strict is required to retain
    # exact FP32 HPR weights when the Qwen backbone loads in BF16.
    _keep_in_fp32_modules_strict = ['hpr']
    # CE and HPR are microbatch means. Trainer divides their sum by accumulation
    # exactly once; forwarding a global token-count only into CE would misweight HPR.
    accepts_loss_kwargs = False

    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = QwenICAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        weight = config.joint['loss_weight']
        if not math.isfinite(weight) or weight < 0 or config.joint['catalog_negatives'] < 0:
            raise ValueError('Finite nonnegative HPR loss weight/negative count required')
        self.hpr = HPRScorer(HPRConfig(config.hidden_size, config.joint['projection_dim'])).float()
        self.post_init()
        self.model.ica.reset_residual()
        self.negative_sampler = None
        semantic_ids = torch.tensor(config.ica['sid_token_ids'][:3], dtype=torch.long)
        self.register_buffer('semantic_token_ids', semantic_ids, persistent=False)
        code = torch.full((config.vocab_size,), -1, dtype=torch.long)
        for ids in config.ica['sid_token_ids']:
            code[ids] = torch.arange(len(ids))
        self.register_buffer('sid_code', code, persistent=False)

    @classmethod
    def from_sft(cls, directory, tokenizer, width, ica, joint, **kwargs):
        base = Qwen2Config.from_pretrained(directory, local_files_only=True)
        if base.model_type != 'qwen2' or getattr(base, 'ica', None):
            raise ValueError('Joint comparison starts from the original ordinary SID SFT, not trained ICA or old HPR')
        if base.vocab_size != len(tokenizer):
            raise ValueError('Do not resize/reset the already trained SFT SID vocabulary')
        values = base.to_dict()
        values.pop('model_type', None)
        config = QwenICAHPRConfig(**values, ica=dict(ica, **tokenizer_binding(tokenizer, width)), joint=joint)
        model = cls.from_pretrained(directory, config=config, local_files_only=True, **kwargs)
        ref = model.get_input_embeddings().weight
        # Explicitly initialize new parameters after HF missing/meta-key handling.
        model.model.ica = ItemContextAttention(config.hidden_size, config.ica).to(ref.device, dtype=ref.dtype)
        model.model.ica.reset_residual()
        model.hpr = HPRScorer(HPRConfig(config.hidden_size, joint['projection_dim'])).to(ref.device).float()
        return model

    def set_catalog(self, catalog):
        codes = np.array([value['sid'][:3] for value in catalog.values()], dtype=np.int64)
        width = self.semantic_token_ids.shape[1]
        if not len(codes) or (codes < 0).any() or (codes >= width).any():
            raise ValueError('Invalid semantic catalog codes')
        self.negative_sampler = PrefixNegativeSampler(codes)

    def answer_locations(self, labels):
        supervised = labels != -100
        starts = supervised.long().argmax(1)
        if (starts < 1).any() or not torch.all(supervised.sum(1) == 5):
            raise ValueError('Joint Q1 expects prompt + four SID tokens + EOS, no packed samples')
        rows = torch.arange(len(labels), device=labels.device)
        columns = starts[:, None] + torch.arange(5, device=labels.device)
        if (columns >= labels.shape[1]).any() or not supervised[rows[:, None], columns].all():
            raise ValueError('Q1 completion must be contiguous')
        ids = labels[rows[:, None], columns[:, :4]]
        if not torch.all(self.model.sid_namespace[ids] == torch.arange(4, device=ids.device)):
            raise ValueError('Completion SID namespaces must be 0,1,2,3')
        if not torch.all(labels[rows, starts + 4] == self.config.eos_token_id):
            raise ValueError('Fifth supervised token must be EOS')
        return starts, self.sid_code[ids[:, :3]]

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                cache_position=None, logits_to_keep=0, **kwargs):
        # One backbone forward shared by CE and HPR, with no all-layer hidden cache.
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            cache_position=cache_position, **kwargs)
        hidden = outputs.last_hidden_state
        selected = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, selected])
        ce = hpr = loss = context = None
        if labels is not None:
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise ValueError('Joint training requires all teacher-forced logits')
            if self.negative_sampler is None:
                raise ValueError('Call set_catalog with the matching prepared SID catalog before training')
            starts, targets = self.answer_locations(labels)
            # Verify label boundaries agree with the real assistant prompt delimiter.
            marker = self.model.assistant_delimiter
            marker_columns = starts[:, None] - len(marker) + torch.arange(len(marker), device=starts.device)
            if (marker_columns < 0).any() or not torch.all(input_ids[torch.arange(len(starts), device=starts.device)[:, None], marker_columns] == marker):
                raise ValueError('Completion labels must start immediately after assistant prompt')
            ce = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            context = hidden[torch.arange(len(starts), device=hidden.device), starts - 1]
            # Current Qwen token embeddings remain in the autograd graph.
            table = self.get_input_embeddings()(self.semantic_token_ids)
            seed = self.config.joint['negative_seed'] + sum(
                (i + 1) * int(v) for i, v in enumerate(targets.detach().cpu().reshape(-1).tolist()))
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                hpr, _ = contrastive_loss(self.hpr, context.float(), targets, table.float(),
                    self.negative_sampler, self.config.joint['catalog_negatives'], np.random.default_rng(seed))
            loss = ce + self.config.joint['loss_weight'] * hpr
        return JointOutput(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            ce_loss=ce, hpr_loss=hpr, prompt_context=context)


AutoConfig.register('qwen2_ica_hpr_joint', QwenICAHPRConfig)
AutoModelForCausalLM.register(QwenICAHPRConfig, QwenICAHPRForCausalLM)
