#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: ShapleyGP.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Compute Shapley effects using the exact or the approximation methods
Reference:
Song et al. (2016) and Goda (2021)
"""
import numpy as np
import itertools
# PyTorch and GPyTorch
import torch
import gpytorch
# BoTorch
from botorch.fit import fit_gpytorch_mll

# Latin Hypercube Sampling
from smt.sampling_methods import LHS

from UQGP.GP import GPutils

torch.set_default_dtype(torch.float64)  # gpytorch by default uses double
# torch.manual_seed(123)


# --------------------------------------------------------------------------- #
# Compute the Shapley effects
# --------------------------------------------------------------------------- #
def compute_shapley_gp(
		train_X, train_y, train_X_bounds,
		N_eval_var_y, N_inner, N_outer,
		exact_or_approx, max_counter, norm_flag=False):
	"""Compute the Shapley values.

	train_X, train_y: Experimental design points (training data)
	train_X_bounds: Lower (train_X_bounds[0]) and Upper (train_X_bounds[1]) bounds
	N_eval_var_y: Number of Monte-Carlo simulations to estimate the total model variance
	N_inner: Number of inner loop
	N_outer: Number of outer loop
	exact_or_approx: Using the exact or approximation method (Song et al. 2016)
	max_counter: Maximum number of permutations
	"""
	# Get the number of inputs
	N_inputs = train_X.shape[-1]

	# Initialize a GP instance
	mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
	# Train the GP model based on train_X and train_y
	fit_gpytorch_mll(mll)  # Using BoTorch's routine

	# Latin hypercube sampling to set the test data points
	sampling = LHS(xlimits=train_X_bounds.T.numpy())
	eval_X_var_y = torch.from_numpy(sampling(N_eval_var_y))

	# Compute the total variance of model output y
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		pred_eval_X = gp.posterior(eval_X_var_y)
		pred_eval_X_mean = pred_eval_X.mean  # Mean of prediction
		# Variance of the predicted outputs
		var_pred_eval_X_mean = torch.var(pred_eval_X_mean)

	def compute_shapley_gp_pi(pi, shap, norm_flag):
		"""Given the permuted set, compute the Shapley values.

		pi: Permuted set
		shap: Shapley values from the previous iteration
		norm_flag: If true, return Shapley values normalized by total variance
		"""
		prevC = 0  # Initialize the previous cost function
		pi_jdxplus = []  # Initialize the index to be fixed

		N_inputs = train_X.shape[-1]  # Number of inputs

		for jdx in range(N_inputs):
			pi_jdx = pi[jdx]
			pi_jdxplus.append(pi_jdx)
			# Setminus to obtain index(s) to be fixed
			minus_pi_jdxplus = np.setdiff1d(np.arange(N_inputs), [pi_jdxplus])

			if (jdx + 1) == N_inputs:
				# No inputs are fixed
				if norm_flag is True:
					c_P_union_pi = 1.
				else:
					c_P_union_pi = var_pred_eval_X_mean
			else:
				#TODO Following Song et al. (2016, Algorithm 1), we take two
				#TODO loops wrt. N_outer and N_inner, which are very slow.
				#TODO We should speed up this part leveraging the gpytorch
				#TODO batch computation.
				#* Maybe DONE

				# Retlive the upper and the lower bounds of inputs
				X_upper_vec = torch.empty([1, len(minus_pi_jdxplus)])
				X_lower_vec = torch.empty_like(X_upper_vec)
				for hdx, hdx_val in enumerate(minus_pi_jdxplus):
					X_lower_vec[:, hdx] = train_X_bounds[0, hdx_val]
					X_upper_vec[:, hdx] = train_X_bounds[1, hdx_val]

				# Draw input that will be fixed
				X_minus_P_piplus = (X_upper_vec - X_lower_vec) * torch.rand(
					N_outer, len(minus_pi_jdxplus)) + X_lower_vec

				mean_pred_X = torch.empty((N_outer, N_inner))
				var_mean_pred_X = torch.empty(N_outer)

				for kdx in range(N_inner):
					# Draw input from the original distribution
					X_P_piplus = (train_X_bounds[1] - train_X_bounds[0]) * torch.rand(
						N_outer, N_inputs) + train_X_bounds[0]
					# Fix inputs with minus_P_piplus
					X_P_piplus[:, minus_pi_jdxplus] = X_minus_P_piplus

					# Make prediction based on posterior
					pred_X = gp.posterior(X_P_piplus)
					mean_pred_X[:, kdx] = pred_X.mean.flatten()

				# Take variance of inner loop, see mean_pred_X
				if norm_flag is True:
					# Normalize by the total variance
					var_mean_pred_X = torch.var(mean_pred_X, axix=1) \
						/ var_pred_eval_X_mean
				else:
					var_mean_pred_X = torch.var(mean_pred_X, axis=1)

				# Take mean of outer loop wrt. pi_jdx
				c_P_union_pi = torch.mean(var_mean_pred_X, axis=0)

			# Update the Shapley value
			delta_c_pi_jdx = c_P_union_pi - prevC
			# Compute the Shapley value
			shap[pi_jdx] += delta_c_pi_jdx

			# Update previous cost
			prevC = c_P_union_pi

		return shap, counter

	shap = np.zeros(N_inputs)  # Initialize the Shapley value
	counter = 0  # Initialize counter

	if exact_or_approx == 'exact':
		perm = itertools.permutations(range(N_inputs), N_inputs)
		for pi in perm:
			shap, counter = compute_shapley_gp_pi(pi, shap, norm_flag)
			counter += 1
	elif exact_or_approx == 'approx':
		while counter < max_counter:
			# Generate one permutation set
			pi = np.random.permutation(N_inputs)
			shap, counter = compute_shapley_gp_pi(pi, shap, norm_flag)
			counter += 1

	return var_pred_eval_X_mean.numpy(), shap / counter