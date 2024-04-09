#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: active_learning.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Active learning acquisition functions used together with Botorch
"""

from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform
from typing import Optional
from torch import Tensor

from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.analytic import PosteriorStandardDeviation
from botorch.acquisition import qNegIntegratedPosteriorVariance

from UQGP.GP import GPutils

class PosteriorVariance(AnalyticAcquisitionFunction):
	r"""Single-outcome Posterior Variance for Active Learning.

	#! BoTorch implemented PosteriorStandardDeviation. It is depreciated.

	This acquisition function quantifies the posterior variance of the model.
	In that, it focuses on pure "exploration" to improve the global
	model accuracy. The acquisition function focuses on the posterior variance
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


class BayesianActiveLearning:
	"""Bayesian active learning to improve an overall GP model accuracy."""

	def __init__(
			self,
			acquisition,  # Acquisition function to be optimized
			N_restarts,  # Number of initial conditions (ICs)
			raw_samples,
			N_MC_samples,
			q_batch):
		self.acquisition = acquisition
		self.N_restarts = N_restarts
		self.raw_samples = raw_samples
		self.N_MC_samples = N_MC_samples
		self.q_batch = q_batch

	def forward(self, train_X, train_y, train_X_bounds):
		"""One step forward active learning"""
		# Instantiate a GP model
		mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
    	# Fit a GP model
		fit_gpytorch_mll(mll)

		if self.acquisition == 'qNIPV':
			# The points to use for MC-integrating the posterior variance
    		# n: The number of (q-batch) samples. As a best practice, it
    		# should be powers of 2
    		# Reshape to have [N, d]
			qmc_samples = draw_sobol_samples(
				bounds=train_X_bounds, n=self.N_MC_samples, q=self.q_batch,
				batch_shape=None, seed=None
    		).squeeze(-2)

    		# Batch Integrated Negative Posterior Variance
    		# mc_points are used for MC-integrating the posterior variance.
			acqf = qNegIntegratedPosteriorVariance(
        		    model=gp,
            		mc_points=qmc_samples
					)
		elif self.acquisition == 'PSTD':
			# Single-outcome posterior standard deviation for active learning
			acqf = PosteriorStandardDeviation(gp)

    	# Optimize the acquisition function
		train_X_add, acq_value = optimize_acqf(
        	acqf, bounds=train_X_bounds, q=self.q_batch,
			num_restarts=self.N_restarts, raw_samples=self.raw_samples,
			# options={"method": "L-BFGS-B"}
			# options={"batch_limit": 10, "init_batch_limit": 512}
        	)
		return train_X_add
