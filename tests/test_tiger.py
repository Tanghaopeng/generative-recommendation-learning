"""Behavioral checks for the local full-SID adapter and search protocol."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from tiger_common import configure, random_state, restore_random
from build_semantic_ids import disambiguate
from train_rqvae import make_model, codec_loss, initialize_codebooks
from tiger_model import TigerModel, pack_histories, training_examples, training_batch


class TigerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(threads=2)

    def fixture(self):
        sid = np.array([[-1]*4, [0,0,0,0], [0,0,0,1], [0,1,0,0],
                        [1,0,1,0], [1,1,1,0], [2,1,0,0]])
        model = TigerModel(sid, dict(vocabulary=4, hidden=16, heads=1, layers=1, ff=32))
        return sid, model

    def test_collision_roundtrip(self):
        raw = np.array([[1,2,3],[1,2,3],[4,5,6],[1,2,3]])
        full, counts = disambiguate(raw)
        self.assertEqual(full[:, -1].tolist(), [0,1,0,2])
        reverse = {tuple(code): item for item, code in enumerate(full, 1)}
        self.assertEqual([reverse[tuple(code)] for code in full], [1,2,3,4])
        self.assertEqual(counts[(1,2,3)], 3)

    def test_upstream_codec_gradients_and_reload(self):
        config = dict(input_dim=8,embed_dim=4,hidden_dims=[16],codebook_size=4,
                      codebook_kmeans_init=False,n_layers=3,n_cat_features=0)
        model = make_model(config)
        x = torch.nn.functional.normalize(torch.randn(48,8),dim=-1)
        initialize_codebooks(model,x,2026,batch_size=16)
        model.train()
        loss,_ = codec_loss(model,x)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.encoder.parameters()),0)
        self.assertTrue(all(layer.embedding.weight.grad is not None for layer in model.layers))
        restored = make_model(config)
        restored.load_state_dict(model.state_dict())
        model.eval(); restored.eval()
        torch.testing.assert_close(model.get_semantic_ids(x).sem_ids,restored.get_semantic_ids(x).sem_ids)

    def test_training_positions_match_sasrec_tail(self):
        rows = [{'train':[1,2,3,4,5]}, {'train':[1,2,3]}]
        examples = training_examples(rows, maxlen=2)
        self.assertEqual(examples.tolist(), [[0,2,3],[0,2,4],[1,0,1],[1,0,2]])
        sid, _ = self.fixture()
        tokens, mask, targets = training_batch(examples, rows, sid, 2)
        self.assertEqual(tokens[0, mask[0]].tolist(), sid[3].tolist())
        self.assertEqual(targets[0].tolist(), sid[4].tolist())
        self.assertEqual(tokens[1, mask[1]].tolist(), sid[[3,4]].reshape(-1).tolist())

    def test_fourth_token_gradient_and_causal_decoder(self):
        sid, model = self.fixture()
        tokens, mask = pack_histories([[1,3],[4]], sid)
        targets = torch.from_numpy(sid[[2,5]])
        model.train()
        model(tokens, mask, targets).backward()
        self.assertGreater(model.decoder_mlp[3].weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            before = model.logits(tokens, mask, targets)
            changed = targets.clone()
            changed[:, 2:] = (changed[:, 2:]+1)%4
            after = model.logits(tokens, mask, changed)
        torch.testing.assert_close(before[:, :3], after[:, :3])

    def test_cached_beam_matches_exhaustive_and_masks_full_history(self):
        sid, model = self.fixture()
        model.eval()
        histories = [[1,3], [4,5]]
        tokens, mask = pack_histories(histories, sid, maxlen=1)
        generated, scores = model.retrieve(tokens, mask, [set(h) for h in histories], beam=20, topk=6)
        for row, history in enumerate(histories):
            t, m = pack_histories([history], sid, maxlen=1)
            candidates = [i for i in range(1,len(sid)) if i not in history]
            with torch.no_grad():
                logits = model.logits(t.repeat(len(candidates),1), m.repeat(len(candidates),1),
                                      torch.from_numpy(sid[candidates]))
                logp = torch.log_softmax(logits,-1)
                actual = logp.gather(-1, torch.from_numpy(sid[candidates]).unsqueeze(-1)).sum((1,2))
            expected = sorted(zip(candidates,actual.tolist()), key=lambda p:(-p[1],p[0]))
            self.assertEqual([i for i in generated[row].tolist() if i], [i for i,_ in expected])
            torch.testing.assert_close(scores[row,:len(expected)], torch.tensor([s for _,s in expected]),
                                       atol=1e-5,rtol=1e-5)
        second, _ = model.retrieve(tokens,mask,[set(h) for h in histories],beam=20,topk=6)
        torch.testing.assert_close(generated,second)
        # Uniform heads should resolve final score ties by ascending item ID.
        for head in model.decoder_mlp:
            torch.nn.init.zeros_(head.weight)
        tied, _ = model.retrieve(tokens,mask,[set(h) for h in histories],beam=20,topk=6)
        self.assertEqual(tied[0].tolist(),[2,4,5,6,0,0])

    def test_optimizer_and_rng_restore_next_update(self):
        sid, model = self.fixture()
        tokens,mask = pack_histories([[1,3],[4]],sid)
        targets = torch.from_numpy(sid[[2,5]])
        optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        def update(net,opt):
            net.train(); opt.zero_grad(); net(tokens,mask,targets).backward(); opt.step()
        update(model,optimizer)
        import copy
        saved = copy.deepcopy((model.state_dict(),optimizer.state_dict()))
        rng = np.random.default_rng(55)
        state = random_state(rng)
        update(model,optimizer)
        expected_draw = rng.permutation(20)
        restored = TigerModel(sid,model.config_local)
        restored.load_state_dict(saved[0])
        other = torch.optim.Adam(restored.parameters(),lr=.001)
        other.load_state_dict(saved[1])
        restore_random(state,rng)
        update(restored,other)
        np.testing.assert_array_equal(expected_draw,rng.permutation(20))
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[name],atol=0,rtol=0)

    def test_vendor_files_unchanged(self):
        info = json.loads((ROOT/'baselines/tiger/upstream_source.json').read_text(encoding='utf-8'))
        for name, expected in info['vendored_files'].items():
            self.assertEqual(hashlib.sha256((ROOT/'baselines/tiger/upstream'/name).read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
