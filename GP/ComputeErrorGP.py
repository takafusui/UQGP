#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: ComputeErrorGP.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Compute the error metrics based on a GP surrogate model
"""
import torch
import gpytorch
# BoTorch
from botorch.fit import fit_gpytorch_mll
# Leave-one-out error
from sklearn.model_selection import LeaveOneOut
# Pre-processing
from sklearn import preprocessing

from UQGP.GP import GPutils

torch_dtype = torch.double


def leave_one_out_GP(train_X, train_y, train_X_bounds):
	"""Compute the leave-one-out error.

	train_X.shape = [n, N_inputs]
	train_y.shape = [n, 1]
	errLOO <= 0.01 is a sufficient accuracy in practice.
	"""
	loo = LeaveOneOut()  # Leave-one-out error estimator instance
	err = torch.empty(loo.get_n_splits(train_X), dtype=torch_dtype)

	# Min-Max scale train_y in [0, 1]
	# Min-Max scaler prepserves the original shape of the distribution of train_y
	# min_max_scaler = preprocessing.MinMaxScaler()
	# train_y_minmax = torch.tensor(min_max_scaler.fit_transform(train_y))

	for train_idx, test_idx in loo.split(train_X):
		train_loo_X, test_loo_X = train_X[train_idx], train_X[test_idx]
		# train_loo_y, test_loo_y \
		#     = train_y_minmax[train_idx], train_y_minmax[test_idx]
		train_loo_y, test_loo_y \
			= train_y[train_idx], train_y[test_idx]

		# Train the GP model based on train_loo_X and train_loo_y
		mll_loo, gp_loo = GPutils.initialize_GP(
			train_loo_X, train_loo_y, train_X_bounds)
		fit_gpytorch_mll(mll_loo)  # Using BoTorch's routine

		# Make prediction
		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			# Posterior prediction
			pred_test_loo_X = gp_loo.posterior(test_loo_X)
			# Compute an squared approximation error
			err[test_idx] = (test_loo_y - pred_test_loo_X.mean)**2

	# Leave-one-out error
	errLOO = 1 / loo.get_n_splits(train_X) * torch.sum(err)

	return errLOO.reshape(1).numpy()


def mean_squared_err_GP(gp, test_X, test_y):
	"""Compute the mean squared error.

	train_X.shape = [n, N_inputs]
	train_y.shape = [n, 1]
	test_X.shape = [m, N_inputs]
	test_y.shape = [m, 1]
	"""
	# Make prediction
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		# Posterior prediction
		pred_test_X = gp.posterior(test_X)
		# Compute mean squared error
		mse = 1 / test_X.shape[0] * torch.sum((test_y - pred_test_X.mean)**2)
	return mse.reshape(1)


def mean_abs_err_GP(gp, test_X, test_y):
	"""Compute the mean absolute error.

	train_X.shape = [n, N_inputs]
	train_y.shape = [n, 1]
	test_X.shape = [m, N_inputs]
	test_y.shape = [m, 1]
	"""
	# Make prediction
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		# Posterior prediction
		pred_test_X = gp.posterior(test_X)
		# Compute mean absolute error
		mae = 1 / test_X.shape[0] * torch.sum(
			torch.abs(test_y - pred_test_X.mean))
	return mae.numpy()


def Q2_GP(gp, test_X, test_y):
	"""Compute the predictively coefficient Q2.

	In practical situations, a metamodel with a predictivity lower than 0.7 is
	often considered as a poor approximation
	"""
	# Make prediction
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		# Posterior prediction
		pred_test_X = gp.posterior(test_X)
	_Q2 = 1 - torch.sum((test_y - pred_test_X.mean)**2) / torch.sum(
		(test_y.mean() - test_y)**2)

	return _Q2
