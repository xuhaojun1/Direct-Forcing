"""Full-token DiT cosine with a staged latent VJP, using the frozen MMD encoder."""

import torch

from losses.cca import token_cosine


class DiTCCALoss(torch.nn.Module):
    requires_embedding = True

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.target_cache = None

    @torch.no_grad()
    def target_tokens(self, target, embedding):
        compute = lambda: self.encoder._forward_tokens(target.detach(), embedding)
        return compute() if self.target_cache is None else self.target_cache.get(target, embedding, compute)

    @torch.no_grad()
    def value(self, prediction, target, embedding):
        return token_cosine(self.encoder._forward_tokens(prediction.detach(), embedding),
                            self.target_tokens(target, embedding)).detach()

    def value_and_gradient(self, prediction, target, embedding):
        with torch.no_grad():
            target_tokens = self.target_tokens(target, embedding)
        with torch.enable_grad():
            leaf = prediction.detach().requires_grad_(True)
            tokens = self.encoder._forward_tokens(leaf, embedding)
            loss = token_cosine(tokens, target_tokens)
            gradient, = torch.autograd.grad(loss, leaf)
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite DiT CCA loss or latent gradient')
        return loss.detach(), gradient.detach().float()
