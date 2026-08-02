"""Framework-independent latent distribution and VAE output types."""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch


class AbstractDistribution:
    """Minimal distribution protocol retained by checkpoint-shaped VAE modules."""

    def sample(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError


class DiracDistribution(AbstractDistribution):
    """Deterministic distribution returning one stored value."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def sample(self) -> torch.Tensor:
        return self.value

    def mode(self) -> torch.Tensor:
        return self.value


class IdentityDistribution(torch.nn.Module):
    """Pass deterministic tokenizer parameters through unchanged."""

    def forward(self, parameters: torch.Tensor):
        zero = parameters.new_zeros(1)
        return parameters, (zero, zero)


class GaussianDistribution(torch.nn.Module):
    """Sample a diagonal Gaussian from concatenated mean/log-variance parameters."""

    def __init__(self, min_logvar: float = -30.0, max_logvar: float = 20.0) -> None:
        super().__init__()
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

    def forward(self, parameters: torch.Tensor):
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        sample = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return sample, (mean, logvar)


class DiagonalGaussianDistribution:
    """Diagonal Gaussian posterior used by native variational autoencoders."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = bool(deterministic)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.std = torch.zeros_like(self.mean)
            self.var = torch.zeros_like(self.mean)

    def sample(
        self,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(
                self.mean.shape,
                generator=generator,
                device=self.mean.device,
                dtype=self.mean.dtype,
            )
        else:
            noise = noise.to(device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self, other: "DiagonalGaussianDistribution | None" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.zeros((), device=self.mean.device, dtype=self.mean.dtype)
        dimensions = tuple(range(1, self.mean.ndim))
        if other is None:
            value = 0.5 * (self.mean.square() + self.var - 1.0 - self.logvar)
        else:
            value = 0.5 * (
                (self.mean - other.mean).square() / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar
            )
        return value.sum(dim=dimensions)

    def nll(self, sample: torch.Tensor, dims: tuple[int, ...] | list[int] | None = None) -> torch.Tensor:
        """Return diagonal-Gaussian negative log likelihood."""

        if self.deterministic:
            return torch.zeros((), device=self.mean.device, dtype=self.mean.dtype)
        dimensions = tuple(dims) if dims is not None else tuple(range(1, self.mean.ndim))
        value = math.log(2.0 * math.pi) + self.logvar + (sample - self.mean).square() / self.var
        return 0.5 * value.sum(dim=dimensions)


def normal_kl(
    mean1: torch.Tensor | float,
    logvar1: torch.Tensor | float,
    mean2: torch.Tensor | float,
    logvar2: torch.Tensor | float,
) -> torch.Tensor:
    """Compute element-wise KL divergence between broadcastable Gaussians."""

    tensor = next(
        (value for value in (mean1, logvar1, mean2, logvar2) if isinstance(value, torch.Tensor)),
        None,
    )
    if tensor is None:
        raise TypeError("normal_kl requires at least one tensor argument")
    first_logvar = logvar1 if isinstance(logvar1, torch.Tensor) else torch.as_tensor(logvar1, device=tensor.device)
    second_logvar = logvar2 if isinstance(logvar2, torch.Tensor) else torch.as_tensor(logvar2, device=tensor.device)
    return 0.5 * (
        -1.0
        + second_logvar
        - first_logvar
        + torch.exp(first_logvar - second_logvar)
        + (mean1 - mean2) ** 2 * torch.exp(-second_logvar)
    )


@dataclass
class AutoencoderKLOutput:
    """Encoded VAE posterior."""

    latent_dist: DiagonalGaussianDistribution


@dataclass
class DecoderOutput:
    """Decoded video tensor and optional posterior."""

    sample: torch.Tensor
    posterior: DiagonalGaussianDistribution | None = None


__all__ = [
    "AbstractDistribution",
    "AutoencoderKLOutput",
    "DecoderOutput",
    "DiagonalGaussianDistribution",
    "DiracDistribution",
    "GaussianDistribution",
    "IdentityDistribution",
    "normal_kl",
]
