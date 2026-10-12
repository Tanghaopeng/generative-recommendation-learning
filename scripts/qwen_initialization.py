"""Reproducible SID initialization shared by Q0 export and Q1 SFT."""
from pathlib import Path

import torch

from qwen_sft import (MODEL_REVISION, MODEL_SHA256, extend_tokenizer, read_json,
                      sha256, sid_tokens)


@torch.no_grad()
def initialize_sid_rows(model, tokenizer, width, seed):
    """Only reset newly assigned SID rows, including pre-existing padded vocab rows."""
    original_size = len(tokenizer)
    original_model_size = model.get_input_embeddings().weight.shape[0]
    vocabulary = extend_tokenizer(tokenizer, width)
    if vocabulary['added_tokens'] != 4 * width:
        raise ValueError('Q0 export requires the original tokenizer without SID tokens')
    ids = [tokenizer.convert_tokens_to_ids(token) for token in sid_tokens(width)]
    if ids != list(range(original_size, len(tokenizer))):
        raise ValueError('SID token IDs must extend the original tokenizer consecutively')
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    inputs = model.get_input_embeddings().weight
    outputs = model.get_output_embeddings().weight
    tied = inputs.data_ptr() == outputs.data_ptr()
    std = model.config.initializer_range
    # Qwen's original model vocab is larger than its tokenizer. Initialize ALL SID
    # rows, not just rows allocated by resize_token_embeddings, so no padded rows
    # silently inherit the base checkpoint's unused embeddings.
    for weight in ([inputs] if tied else [inputs, outputs]):
        values = torch.randn((len(ids), weight.shape[1]), generator=generator,
                             dtype=torch.float32) * std
        weight.index_copy_(0, torch.tensor(ids, device=weight.device), values.to(weight))
    model.config.pad_token_id = tokenizer.pad_token_id
    return dict(vocabulary=vocabulary, original_model_vocab_size=original_model_size,
                original_tokenizer_size=original_size, sid_rows_initialized=len(ids),
                reused_padding_rows=max(0, min(original_model_size, len(tokenizer)) - original_size),
                input_output_embeddings_tied=tied, initializer_std=std,
                initialization='CPU float32 N(0, initializer_range), cast to weight dtype', seed=seed)


def check_initialization(directory, data, seed=None):
    """Reject a mismatched/corrupted Q0 before training or evaluation."""
    directory, data = Path(directory), Path(data)
    record = read_json(directory / 'q0_initialization.json')
    if (record.get('status') != 'q0_export_completed' or record.get('optimizer_steps') != 0 or
            record.get('source_model_sha256') != MODEL_SHA256 or record.get('model_revision') != MODEL_REVISION):
        raise ValueError('Not a completed Q0 export from the pinned original model')
    if record['prepared_manifest_sha256'] != sha256(data / 'manifest.json'):
        raise ValueError('Q0 data/SID manifest differs from the requested data')
    if seed is not None and record['seed'] != seed:
        raise ValueError('Q0 initialization seed differs from the SFT seed')
    for name, fingerprint in record['files_sha256'].items():
        path = directory / name
        if path.parent != directory or sha256(path) != fingerprint:
            raise ValueError('Q0 artifact checksum mismatch: ' + name)
    if record['checkpoint_weight_sha256'] != record['files_sha256']['model.safetensors']:
        raise ValueError('Q0 weight fingerprint is inconsistent')
    return record
