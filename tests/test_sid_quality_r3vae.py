import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'scripts'))
from sid_quality import cluster_quality, semantic_quality
from r3vae import R3VAE, pd_codebook_loss


class QualityTests(unittest.TestCase):
    def test_sc_matches_explicit_pair_average(self):
        x = np.array([[1.,0.],[1.,1.],[0.,1.],[-1.,0.]])
        labels = np.array([0,0,0,1])
        unit = x / np.linalg.norm(x,axis=1,keepdims=True)
        oracle = np.mean([unit[a] @ unit[b] for a,b in itertools.combinations(range(3),2)])
        result = cluster_quality(x,labels)
        self.assertAlmostEqual(result['SC'],oracle)
        self.assertEqual(result['singleton_clusters'],1)
        self.assertEqual(result['sc_item_coverage'],.75)

    def test_pd_matches_two_cluster_geometry(self):
        x = np.array([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        result = cluster_quality(x,np.array([0,0,1,1]))
        self.assertAlmostEqual(result['SC'],1)
        self.assertAlmostEqual(result['PD'],-2)
        self.assertEqual(result['pd_total_pairs'],1)

    def test_singletons_and_zero_centroids_are_explicit(self):
        result = cluster_quality(np.eye(3),np.arange(3))
        self.assertIsNone(result['SC'])
        self.assertEqual(result['sc_item_coverage'],0)
        result = cluster_quality(np.array([[1.,0.],[-1.,0.]]),np.array([0,0]))
        self.assertIsNone(result['PD'])
        self.assertEqual(result['pd_undefined_centroids'],1)

    def test_sampling_is_deterministic_and_bounded(self):
        x = np.random.default_rng(3).normal(size=(100,8))
        first = cluster_quality(x,np.arange(100),max_pairs=123,seed=42)
        self.assertEqual(first,cluster_quality(x,np.arange(100),max_pairs=123,seed=42))
        self.assertFalse(first['pd_exact'])
        self.assertTrue(-4 <= first['PD'] <= 0)
        self.assertEqual(first['pd_evaluated_pairs'],123)

    def test_no_disambiguation_as_semantics(self):
        x = np.eye(3)
        with self.assertRaises(ValueError):
            semantic_quality(x,np.zeros((3,4),dtype=int))
        with self.assertRaises(ValueError):
            cluster_quality(np.zeros((2,3)),np.array([0,0]))


class R3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self,mode='official_top1'):
        torch.manual_seed(123)
        return R3VAE(input_dim=8,latent_dim=4,hidden_dims=[12],codebook_size=4,rating_mode=mode)

    def test_reference_projection_is_orthogonal(self):
        model = self.model()
        residual, anchor = model.initial_residual(torch.randn(7,4))
        self.assertLess(float((residual * model.reference).sum(-1).abs().max().detach()),1e-6)

    def test_reconstruction_updates_reference_encoder_decoder_codebooks(self):
        model = self.model()
        out = model(torch.randn(12,8))
        out['loss'].backward()
        for name,parameters in [('encoder',model.encoder.parameters()),('decoder',model.decoder.parameters()),
                                ('reference',[model.reference]),('codebooks',model.codebooks)]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None),0,name)
        self.assertEqual(tuple(out['ids'].shape),(12,3))

    def test_optional_loss_weights_preserve_forward_and_change_gradients(self):
        model = self.model()
        x = torch.randn(12,8)
        first = model(x)
        second = model(x,.1,.05)
        torch.testing.assert_close(first['ids'],second['ids'])
        torch.testing.assert_close(first['reconstruction_loss'],second['reconstruction_loss'])
        torch.testing.assert_close(second['loss'],first['loss']+.1*first['sc_proxy_loss']+.05*first['pd_proxy_loss'])
        grad = torch.autograd.grad(first['loss'],model.codebooks[0],retain_graph=True)[0]
        reg_grad = torch.autograd.grad(second['loss'],model.codebooks[0])[0]
        self.assertGreater(float((grad-reg_grad).abs().sum()),0)

    def test_softmax_mode_has_dense_codebook_reconstruction_gradient(self):
        model = self.model('paper_softmax')
        model(torch.randn(9,8))['loss'].backward()
        for codebook in model.codebooks:
            self.assertTrue((codebook.grad.abs().sum(-1)>0).all())

    def test_pd_matches_reference_formula_and_has_gradient(self):
        x = torch.randn(7,4,requires_grad=True)
        unit = F.normalize(x,dim=-1)
        matrix = unit @ unit.T
        expected = torch.exp(-2*(1-matrix[~torch.eye(7,dtype=torch.bool)])).mean().log()
        torch.testing.assert_close(pd_codebook_loss(x),expected)
        pd_codebook_loss(x).backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_kmeans_uses_train_sample_and_is_repeatable(self):
        x = torch.randn(40,8)
        first, second = self.model(), self.model()
        first.initialize_kmeans(x,42)
        second.initialize_kmeans(x,42)
        for left,right in zip(first.codebooks,second.codebooks):
            torch.testing.assert_close(left,right,atol=0,rtol=0)

    def test_quantizer_matches_pinned_public_implementation(self):
        path = ROOT / 'outputs/vendor_cache/r3vae_official_20261016/src/models/r3_vae.py'
        if not path.is_file():
            self.skipTest('Optional pinned upstream checkout not downloaded')
        spec = importlib.util.spec_from_file_location('r3vae_pinned_oracle',path)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        model = self.model()
        oracle = upstream.Codebook(4,3,['4']*3)
        with torch.no_grad():
            for depth,codebook in enumerate(model.codebooks):
                getattr(oracle,'codebook_'+str(depth)).copy_(codebook)
        encoded = torch.randn(12,4,requires_grad=True)
        residual, anchor = model.initial_residual(encoded)
        actual = model.quantize_encoded(encoded)
        quantized,ids,sc,pd = oracle(residual,quants=anchor)
        for left,right in [(actual['quantized'],quantized),(actual['ids'],ids),
                           (actual['sc_proxy_loss'],sc),(actual['pd_proxy_loss'],pd)]:
            torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-5)
        ag = torch.autograd.grad(actual['quantized'].square().sum()+actual['sc_proxy_loss']+actual['pd_proxy_loss'],encoded,retain_graph=True)[0]
        og = torch.autograd.grad(quantized.square().sum()+sc+pd,encoded)[0]
        torch.testing.assert_close(ag,og,atol=1e-6,rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
