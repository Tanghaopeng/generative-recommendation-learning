import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from tiger_common import configure,random_state,restore_random
from train_rqvae import make_model,codec_loss
from rqvae_ema import EmaQuantize,attach_ema,ema_step,canonical_state
from audit_codebooks import occupancy,audit_gradients


class EmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(threads=2)

    def fixture(self):
        config=dict(input_dim=8,embed_dim=4,hidden_dims=[16],codebook_size=4,
                    codebook_kmeans_init=False,n_layers=3,n_cat_features=0,commitment_weight=.25)
        model=make_model(config)
        keys=set(model.state_dict())
        return config,keys,attach_ema(model,decay=.9)

    def test_ema_update_matches_counts_and_sums(self):
        weight=torch.tensor([[0.,0.],[10.,10.],[20.,20.]])
        layer=EmaQuantize(weight,decay=.5)
        x=torch.tensor([[1.,1.],[2.,2.],[9.,9.]],requires_grad=True)
        out=layer(x)
        torch.testing.assert_close(layer.weight,weight)
        out.loss.mean().backward()
        self.assertIsNone(layer.weight.grad)
        self.assertGreater(x.grad.abs().sum().item(),0)
        layer.ema_step()
        # code0 N=.5*1+.5*2=1.5; M=0+.5*(1+2)=1.5 =>1
        # code1 N=1; M=.5*10+.5*9=9.5; code2 unchanged
        torch.testing.assert_close(layer.weight,torch.tensor([[1.,1.],[9.5,9.5],[20.,20.]]))

    def test_eval_cannot_update_codebook(self):
        _,_,model=self.fixture()
        model.eval()
        before=copy.deepcopy(model.state_dict())
        model.get_semantic_ids(torch.randn(20,8))
        for key,value in before.items():
            torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)
        with self.assertRaises(ValueError):
            ema_step(model)

    def test_export_matches_original_inference_and_does_not_draw_rng(self):
        config=dict(input_dim=8,embed_dim=4,hidden_dims=[16],codebook_size=4,
                    codebook_kmeans_init=False,n_layers=3,n_cat_features=0)
        base=make_model(config)
        keys=set(base.state_dict())
        before=torch.get_rng_state().clone()
        attach_ema(base)
        torch.testing.assert_close(before,torch.get_rng_state(),atol=0,rtol=0)
        original=make_model(config)
        original.load_state_dict(canonical_state(base,keys),strict=True)
        x=torch.randn(20,8)
        base.eval(); original.eval()
        torch.testing.assert_close(base.get_semantic_ids(x).sem_ids,original.get_semantic_ids(x).sem_ids)
        torch.testing.assert_close(base.get_semantic_ids(x).embeddings,original.get_semantic_ids(x).embeddings)

    def test_full_ema_optimizer_rng_resume_next_step(self):
        _,_,model=self.fixture()
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        x=torch.nn.functional.normalize(torch.randn(20,8),dim=-1)
        def step(net,opt):
            net.train(); opt.zero_grad(); codec_loss(net,x)[0].backward(); opt.step(); ema_step(net)
        step(model,optimizer)
        saved=copy.deepcopy((model.state_dict(),optimizer.state_dict()))
        rng=np.random.default_rng(2026)
        random=random_state(rng)
        step(model,optimizer)
        draw=rng.permutation(20)
        _,_,other=self.fixture()
        other.load_state_dict(saved[0])
        opt=torch.optim.Adam([p for p in other.parameters() if p.requires_grad],lr=.001)
        opt.load_state_dict(saved[1])
        restore_random(random,rng)
        step(other,opt)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,other.state_dict()[key],atol=0,rtol=0)
        np.testing.assert_array_equal(draw,rng.permutation(20))

    def test_effective_usage_detects_skew_despite_full_occupancy(self):
        uniform=occupancy([25,25,25,25])
        skewed=occupancy([97,1,1,1])
        self.assertEqual(skewed['dead_codes'],0)
        self.assertEqual(skewed['utilization'],1)
        self.assertAlmostEqual(uniform['effective_codes'],4)
        self.assertLess(skewed['effective_codes'],1.3)

    def test_inherited_residual_commitment_gradient_routing(self):
        _,_,model=self.fixture()
        x=torch.nn.functional.normalize(torch.randn(24,8),dim=-1)
        gradients=audit_gradients(model,x)['layers']
        self.assertGreater(gradients[0]['commitment_loss_encoder_gradient'],0)
        self.assertEqual([r['commitment_loss_encoder_gradient'] for r in gradients[1:]],[0.,0.])
        self.assertTrue(all(r['codebook_loss_codebook_gradient']==0 for r in gradients))


if __name__=='__main__':
    unittest.main()
