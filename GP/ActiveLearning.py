#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: active_learning.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Active learning acquision functions used together with Botorch
"""

from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform
from typing import Optional
from torch import Tensor


class PosteriorVariance(AnalyticAcquisitionFunction):
    r"""Single-outcome Posterior Variance for Active Learning.

    This acquisition function quantifies the posterior variance of the model.
    In that, it focuses on pure "exploration" to improve the global
    model accurety. The acquasition function focuses on the posterior variance
    at the query points. Only supports the case of `q=1` (i.e. greedy,
    non-batch selection of design points). The model must be single-outcome.

    `MSE(x) = var(x)`, where 'var` is the posterior variance.
    """

    def __init__(
        self,
        model: Model,
        posterior_transform: Optional[PosteriorTransform] = None,
        **kwargs,
    ) -> None:
        r"""Single-outcome mean-square error.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of MSE values at the given design points `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean
        view_shape = mean.shape[:-2] if mean.shape[-2] == 1 else mean.shape[:-1]
        variance = posterior.variance.view(view_shape)

        return variance
