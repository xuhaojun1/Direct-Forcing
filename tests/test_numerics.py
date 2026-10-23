import unittest
import numpy as np
import torch
from references.bank import RDMReference
from losses.rdm import mmd_nystrom_with_terms
from references.tokens import build_token_reference, pool_cost_matrix
from losses.cca import CCAFeatureLoss, token_cosine


class Encoder(torch.nn.Module):
    def _forward_tokens(self, x):
        return x.permute(0, 2, 3, 4, 1).flatten(1, 3).square()


class Decoder(torch.nn.Module):
    def decode_value(self, x):
        return x.tanh()

    def full_vjp(self, x, gradient):
        x = x.detach().requires_grad_(True)
        return torch.autograd.grad(self.decode_value(x), x, gradient)[0]


class NumericsTests(unittest.TestCase):
    def test_staged_cca_vjp_matches_direct_autograd(self):
        torch.manual_seed(8)
        prediction = (torch.randn(2, 3, 4, 2, 2) * .2).requires_grad_(True)
        target = torch.randn_like(prediction) * .2
        objective = CCAFeatureLoss(Encoder(), Decoder())
        value = objective(prediction, target)
        staged, = torch.autograd.grad(value, prediction)
        pixels = objective.decoder.decode_value(prediction.transpose(1, 2))
        direct_loss = token_cosine(objective._tokens(pixels), objective.target_tokens(target))
        direct, = torch.autograd.grad(direct_loss, prediction)
        torch.testing.assert_close(value, direct_loss)
        torch.testing.assert_close(staged, direct, atol=1e-7, rtol=1e-5)

    def test_small_full_landmark_reference_matches_empirical_mmd_gradient(self):
        torch.manual_seed(7)
        reference = torch.randn(6, 4)
        generated = torch.randn(3, 4, requires_grad=True)
        sigma = 2.
        kernel = lambda a, b: torch.exp(-torch.cdist(a, b).square() / (2 * sigma**2))
        krr = float(kernel(reference, reference).mean())
        bundle = build_token_reference(reference.numpy(), sigma, n_landmarks=6,
            device='cpu', k_rr=krr, jitter=1e-8, block=2)
        z = bundle['Z']
        bank = RDMReference(z, z.square().sum(1), bundle['alpha'], sigma, krr, {}, '')
        actual, _ = mmd_nystrom_with_terms(generated, bank, gen_chunk=2)
        expected = kernel(generated, generated).mean() - 2 * kernel(generated, reference).mean() + krr
        ag, = torch.autograd.grad(actual, generated, retain_graph=True)
        eg, = torch.autograd.grad(expected, generated)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(ag, eg, atol=2e-6, rtol=2e-5)
        distances = pool_cost_matrix(reference.numpy(), device='cpu', block=2)
        torch.testing.assert_close(distances, torch.cdist(reference, reference).square(), atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
