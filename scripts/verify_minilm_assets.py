"""Verify packaged MiniLM files and a small offline frozen inference pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / '02_商品编码/01_MiniLM'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encode(model, tokenizer, texts):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
        vector = F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), p=2, dim=1)
    return batch, vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare-cache', type=Path, help='Optional original complete HF model directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((BASE / '来源/manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path = BASE / record['path']
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'Packaged file changed: {record["path"]}')
    tokenizer = AutoTokenizer.from_pretrained(BASE / '分词器', local_files_only=True)
    model = AutoModel.from_pretrained(BASE / '模型权重', local_files_only=True).eval().requires_grad_(False)
    texts = ['title: Blue ballpoint pen. category: Office Products.',
             'title: A4 printer paper. brand: Example. description: White paper for office printing.']
    batch, vectors = encode(model, tokenizer, texts)
    assert vectors.shape == (2, 384) and torch.isfinite(vectors).all()
    torch.testing.assert_close(torch.linalg.vector_norm(vectors, dim=1), torch.ones(2))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert parameters == 22713216 and len(tokenizer) == model.config.vocab_size == 30522
    comparison = None
    if args.compare_cache:
        old_tokenizer = AutoTokenizer.from_pretrained(args.compare_cache, local_files_only=True)
        old_model = AutoModel.from_pretrained(args.compare_cache, local_files_only=True).eval().requires_grad_(False)
        old_batch, old_vectors = encode(old_model, old_tokenizer, texts)
        for key in batch:
            torch.testing.assert_close(batch[key], old_batch[key], rtol=0, atol=0)
        torch.testing.assert_close(vectors, old_vectors, rtol=0, atol=0)
        comparison = dict(token_ids_equal=True, vectors_bitwise_equal=True, max_absolute_difference=0.0)
    result = dict(status='passed', date='2026-10-09', model_id=manifest['model_id'], revision=manifest['revision'],
                  files_verified=len(manifest['files']), model_class=type(model).__name__, parameters=parameters,
                  vocabulary=len(tokenizer), layers=model.config.num_hidden_layers,
                  attention_heads=model.config.num_attention_heads, hidden_size=model.config.hidden_size,
                  intermediate_size=model.config.intermediate_size,
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  output_shape=list(vectors.shape), l2_norms=torch.linalg.vector_norm(vectors, dim=1).tolist(),
                  max_tokens=128, local_files_only=True, original_cache_comparison=comparison,
                  scope='two synthetic product texts; offline inference only; no training or recommendation metrics')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(result)


if __name__ == '__main__':
    main()
