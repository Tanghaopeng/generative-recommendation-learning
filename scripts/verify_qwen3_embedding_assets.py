"""Bounded offline verification: hashes, safetensors headers and tokenizer; no 4B forward."""
from pathlib import Path
import hashlib
import json
import math

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '02_商品编码/02_Qwen3_Embedding_4B'


def main():
    manifest = json.loads((BASE/'来源/manifest.json').read_text(encoding='utf-8'))
    folder = BASE/'模型文件'
    for record in manifest['files']:
        path = folder/record['path']
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                h.update(block)
        assert path.stat().st_size == record['bytes'] and h.hexdigest() == record['sha256']
    cfg = AutoConfig.from_pretrained(folder, local_files_only=True)
    assert cfg.model_type == 'qwen3' and cfg.hidden_size == 2560 and cfg.num_hidden_layers == 36
    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, padding_side='left')
    batch = tok(['A4 printer paper.', 'Blue ballpoint pen for office writing.'],
                padding=True, truncation=True, max_length=128, return_tensors='pt')
    assert batch['attention_mask'][:, -1].all() and batch['input_ids'].max() < cfg.vocab_size
    pool = json.loads((folder/'1_Pooling/config.json').read_text())
    assert pool['pooling_mode_lasttoken'] and pool['word_embedding_dimension'] == cfg.hidden_size
    index = json.loads((folder/'model.safetensors.index.json').read_text())
    found, parameters, weight_bytes = {}, 0, 0
    for filename in sorted(set(index['weight_map'].values())):
        with safe_open(folder/filename, framework='pt', device='cpu') as handle:
            for key in handle.keys():
                assert key not in found
                shape = handle.get_slice(key).get_shape()
                assert handle.get_slice(key).get_dtype() == 'BF16'
                found[key] = filename
                parameters += math.prod(shape)
                weight_bytes += math.prod(shape)*2
    assert found == index['weight_map'] and weight_bytes == index['metadata']['total_size']
    result = {'status': 'passed', 'model_id': manifest['model_id'], 'revision': manifest['revision'],
              'files_verified': len(manifest['files']), 'stored_parameters': parameters,
              'weight_tensor_bytes': weight_bytes, 'layers': cfg.num_hidden_layers,
              'hidden_size': cfg.hidden_size, 'embedding_dimension': cfg.hidden_size,
              'attention_heads': cfg.num_attention_heads, 'kv_heads': cfg.num_key_value_heads,
              'head_dim': cfg.head_dim, 'tokenizer_batch_shape': list(batch['input_ids'].shape),
              'pooling': 'last effective token + L2 normalization', 'local_files_only': True,
              'scope': 'SHA256, safetensors headers/index and tokenizer only; no full-model forward, training or catalog embedding'}
    (BASE/'来源/离线验证.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
