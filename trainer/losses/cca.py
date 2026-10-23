"""CCA in native-video feature space with staged first-order input VJPs."""

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def token_cosine(prediction, target):
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("CCA features require identical [batch,tokens,channels] grids")
    return (1 - F.cosine_similarity(prediction.float(), target.detach().float(), dim=-1)).mean()


class _CCALatentLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, target, objective):
        value, gradient = objective.value_and_gradient(prediction.detach(), target.detach())
        ctx.save_for_backward(gradient)
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        gradient, = ctx.saved_tensors
        return gradient * upstream.to(gradient), None, None


class CCAFeatureLoss(torch.nn.Module):
    """Keep the encoder's complete token grid; temporal pairing is supplied by CCA."""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder.eval().requires_grad_(False)
        self.decoder = decoder.eval().requires_grad_(False)
        self.contract = {}
        self.target_cache = None

    def train(self, mode=True):
        super().train(False)
        return self

    def _tokens(self, pixels):
        return self.encoder._forward_tokens((pixels.float() * .5 + .5).clamp(0, 1))

    feature_loss = staticmethod(token_cosine)

    @torch.no_grad()
    def target_tokens(self, target):
        compute = lambda: self._tokens(self.decoder.decode_value(target.detach().transpose(1, 2)))
        return compute() if self.target_cache is None else self.target_cache.get(target, None, compute)

    @torch.no_grad()
    def value(self, prediction, target):
        target_tokens = self.target_tokens(target)
        prediction_tokens = self._tokens(self.decoder.decode_value(prediction.detach().transpose(1, 2)))
        return self.feature_loss(prediction_tokens, target_tokens).detach()

    def value_and_gradient(self, prediction, target):
        """Free the encoder's graph before building the decoder input-gradient graph."""
        prediction_tc = prediction.detach().transpose(1, 2)
        with torch.no_grad():
            target_tokens = self.target_tokens(target)
            decoded = self.decoder.decode_value(prediction_tc)
        with torch.enable_grad():
            pixels = decoded.detach().requires_grad_(True)
            tokens = self._tokens(pixels)
            loss = self.feature_loss(tokens, target_tokens)
            pixel_gradient, = torch.autograd.grad(loss, pixels)
            value = loss.detach()
        del loss, tokens, target_tokens, decoded, pixels
        with torch.enable_grad():
            gradient = self.decoder.full_vjp(prediction_tc, pixel_gradient.detach())
        gradient = gradient.transpose(1, 2).contiguous().float()
        if not torch.isfinite(value).all() or not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite CCA feature value or latent gradient")
        return value, gradient

    def forward(self, prediction, target):
        if torch.is_grad_enabled() and prediction.requires_grad:
            return _CCALatentLoss.apply(prediction, target, self)
        return self.value(prediction, target)
