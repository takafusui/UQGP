#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: UnivariateGP.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Compute univariate effects using a GP surrogate model
Reference:
Younes et al. (2013)
"""
import numpy as np
import torch
import gpytorch
# BoTorch
from botorch.fit import fit_gpytorch_mll

from UQGP.GP import GPutils


def compute_univariate_pred(train_X, train_y, train_X_bounds, test_X, N_Xi):
	"""Compute the univariate effect using the GP prediction only.

	n: Number of experimental design
	m: Number of test points
	train_X.shape = [n, N_inputs]
	train_y.shape = [n, 1]
	test_X.shape = [m, N_inputs]

	train_X, train_y: Experimental design points (training data)
	train_X_bounds: Lower (train_X_bounds[0]) and Upper (train_X_bounds[1]) bounds

	return
	uni_pred.shape = [N_inputs, N_Xi]
	"""
	# Get the dimensions of train_X and train_y
	# train_X_dim = train_X.shape[-1]
	# train_y_dim = train_y.shape[-1]

	# Get the number of inputs
	N_inputs = train_X.shape[-1]
	# Number of test points
	N_test_X = test_X.shape[0]

	# Train the GP model based on the standardized (train_X and train_y)
	# likelihood = gpytorch.likelihoods.GaussianLikelihood()
	# gp = SingleTaskGP(
	#     train_X, train_y,  # likelihood=likelihood,
	#     input_transform=InputStandardize(d=train_X_dim),
	#     outcome_transform=Standardize(m=train_y_dim)
	# )
	# mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
	# xlimits = np.stack((X_lower, X_upper), axis=1)
	# train_X_bounds = torch.from_numpy(xlimits.T)

	# Initialize a GP model
	mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
	# Train the GP model based on train_X and train_y
	fit_gpytorch_mll(mll)  # Using BoTorch's routine

	# Univariate effect using GP prediction
	uni_pred = torch.empty([N_inputs, N_Xi])

	for idx in range(N_inputs):
		# Fix the model parameter i that is evenly spaced
		Xi = torch.linspace(
			train_X_bounds[0, idx], train_X_bounds[1, idx], N_Xi)

		# Store the mean of the mean of predictions
		pred_test_Xi_mean = torch.empty([N_test_X, N_Xi])

		for jdx in range(N_Xi):
			# Fix the parameter idx at Xi[jdx]
			test_Xi = test_X.detach().clone()
			test_Xi[:, idx] = Xi[jdx]

			# Draw N_sampling samples from the trained GP model
			# Use torch.distributions.Normal() method
			with torch.no_grad(), gpytorch.settings.fast_pred_var():
				pred_Xi = gp.posterior(test_Xi)
				pred_Xi_mean = pred_Xi.mean  # Prediction mean
			pred_test_Xi_mean[:, jdx] = pred_Xi_mean[0]

			# import ipdb; ipdb.set_trace()
		# Take the mean over X1, X2, ..., Xd (Xi.shape=N_test_X)
		uni_pred[idx, :] = torch.mean(pred_test_Xi_mean, axis=0)

		# import ipdb; ipdb.set_trace()
	return uni_pred


def compute_univariate_tilde(
		train_X, train_y, train_X_bounds, test_X, N_Xi, N_sampling):
	"""Compute the univariate effect using the global GP model.

	n: Number of experimental design
	m: Number of test points
	train_X.shape = [n, N_inputs]
	train_y.shape = [n, 1]
	test_X.shape = [m, N_inputs]

	train_X, train_y: Experimental design points (training data)
	train_X_bounds: Lower (train_X_bounds[0]) and Upper (train_X_bounds[1]) bounds

	return
	mu_E_Ytilde_Xi.shape = [N_inputs, N_Xi]
	sigma_E_Ytilde_Xi.shape = [N_inputs, N_Xi]
	"""
	# Get the dimensions of train_X and train_y
	# train_X_dim = train_X.shape[-1]
	# train_y_dim = train_y.shape[-1]

	# Get the number of inputs
	N_inputs = train_X.shape[-1]
	# Number of test points
	N_test_X = test_X.shape[0]

	# Train the GP model based on the standardized (train_X and train_y)
	# likelihood = gpytorch.likelihoods.GaussianLikelihood()
	# gp = SingleTaskGP(
	#     train_X, train_y,  # likelihood=likelihood,
	#     input_transform=InputStandardize(d=train_X_dim),
	#     outcome_transform=Standardize(m=train_y_dim)
	# )
	# mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
	# xlimits = np.stack((X_lower, X_upper), axis=1)
	# train_X_bounds = torch.from_numpy(xlimits.T)

	# Initialize a GP model
	mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
	# Train the GP model based on train_X and train_y
	fit_gpytorch_mll(mll)  # Using BoTorch's routine

	Ytilde_Xi = torch.empty([N_sampling, N_test_X, N_Xi])
	E_Ytilde_Xi = torch.empty([N_inputs, N_sampling, N_Xi])

	for idx in range(N_inputs):
		# Fix the model parameter i that is evenly spaced
		Xi = torch.linspace(
			train_X_bounds[0, idx], train_X_bounds[1, idx], N_Xi)

		for jdx in range(N_Xi):
			# Fix the parameter idx at Xi[jdx]
			test_Xi = test_X.detach().clone()
			test_Xi[:, idx] = Xi[jdx]

			# Draw N_sampling samples from the trained GP model
			# Use torch.distributions.Normal() method
			with torch.no_grad(), gpytorch.settings.fast_pred_var():
				pred_test_Xi = gp.posterior(test_Xi)
				Ytilde_Xi[:, :, jdx] = torch.distributions.Normal(
					pred_test_Xi.mean[:, 0],
					torch.sqrt(pred_test_Xi.variance[:, 0])
				).sample(sample_shape=torch.Size([N_sampling, ]))

		# Take the mean over X1, X2, ..., Xd (Xi.shape=N_test_X)
		E_Ytilde_Xi[idx, :, :] = torch.mean(Ytilde_Xi, axis=1)

	# Take the mean and the standard deviation wrt. N_sampling
	mu_E_Ytilde_Xi = torch.mean(E_Ytilde_Xi, axis=1)
	sigma_E_Ytilde_Xi = torch.std(E_Ytilde_Xi, axis=1)

	return mu_E_Ytilde_Xi, sigma_E_Ytilde_Xi
