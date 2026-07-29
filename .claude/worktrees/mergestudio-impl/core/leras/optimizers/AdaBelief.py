import math
from typing import Iterable, List, Optional

import numpy as np
import torch

from core.leras import nn


def _torch_to_numpy_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return np.float16
    if dtype == torch.float32:
        return np.float32
    if dtype == torch.float64:
        return np.float64
    return np.float32


class AdaBelief(nn.OptimizerBase):
    """DeepFaceLab AdaBelief implementation (TF-equivalent math)."""

    def __init__(
        self,
        trainable_weights: Optional[Iterable] = None,
        lr: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        lr_dropout: float = 1.0,
        lr_cos: int = 0,
        clipnorm: float = 0.0,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)

        if name is None:
            self.name = self.__class__.__name__

        self.lr = float(lr)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.lr_dropout = float(lr_dropout)
        self.lr_cos = int(lr_cos)
        self.clipnorm = float(clipnorm)

        self.iterations = torch.tensor(0, dtype=torch.int64)

        self.params: List[torch.nn.Parameter] = []
        self.ms: List[torch.Tensor] = []
        self.vs: List[torch.Tensor] = []
        self.lr_masks: List[Optional[torch.Tensor]] = []

        if trainable_weights is not None:
            self.initialize_variables(trainable_weights)

    def get_weights(self):
        return [self.iterations] + self.ms + self.vs

    def initialize_variables(self, trainable_weights, vars_on_cpu=True, lr_dropout_on_cpu=False):
        params: List[torch.nn.Parameter] = []
        for item in trainable_weights:
            if isinstance(item, (list, tuple)):
                for p in item:
                    if isinstance(p, torch.nn.Parameter):
                        params.append(p)
            elif isinstance(item, torch.nn.Parameter):
                params.append(item)
            elif hasattr(item, 'parameters'):
                params.extend([p for p in item.parameters() if isinstance(p, torch.nn.Parameter)])

        self.params = params
        self.ms = [torch.zeros_like(p, device=p.device, dtype=p.dtype) for p in self.params]
        self.vs = [torch.zeros_like(p, device=p.device, dtype=p.dtype) for p in self.params]

        if self.lr_dropout != 1.0:
            self.lr_masks = [
                torch.bernoulli(torch.full_like(p, self.lr_dropout, dtype=p.dtype, device=p.device))
                for p in self.params
            ]
        else:
            self.lr_masks = [None for _ in self.params]

    def _get_lr_value(self) -> float:
        lr = self.lr
        if self.lr_cos != 0:
            t = float(int(self.iterations.item()))
            lr *= (math.cos(t * (2 * math.pi / float(self.lr_cos))) + 1.0) / 2.0
        return lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        if len(self.params) == 0:
            return

        self.iterations += 1
        lr = self._get_lr_value()

        scale = 1.0
        if self.clipnorm > 0.0:
            sq_sum = None
            for p in self.params:
                if p.grad is None:
                    continue
                g = p.grad
                v = (g.detach().float() ** 2).sum()
                sq_sum = v if sq_sum is None else (sq_sum + v)
            if sq_sum is not None:
                norm = torch.sqrt(sq_sum)
                n = float(norm.item())
                if n >= self.clipnorm and n > 0.0:
                    scale = float(self.clipnorm) / n

        b1 = self.beta_1
        b2 = self.beta_2

        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                g = p.grad
                if scale != 1.0:
                    g = g * scale

                m = self.ms[i]
                v = self.vs[i]

                m_t = b1 * m + (1.0 - b1) * g
                v_t = b2 * v + (1.0 - b2) * (g - m_t).pow(2)

                resolution = float(np.finfo(_torch_to_numpy_dtype(p.dtype)).resolution)
                step_delta = (-lr) * m_t / (torch.sqrt(v_t) + resolution)

                mask = self.lr_masks[i]
                if mask is not None:
                    step_delta = step_delta * mask

                p.add_(step_delta)
                self.ms[i] = m_t
                self.vs[i] = v_t


nn.AdaBelief = AdaBelief

