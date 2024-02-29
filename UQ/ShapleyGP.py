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

torch.set_default_dtype(torch.float64)
# torch.manual_seed(123)


# --------------------------------------------------------------------------- #
# Compute the Shapley effects
# --------------------------------------------------------------------------- #
def compute_shapley_gp(
        train_X, train_y, train_X_bounds,
        N_eval_var_y, N_test_X, N_Xi,
        exact_or_approx, max_counter, norm_flag=False):
    """Compute the Shapley values.

    train_X, train_y: Experimental design points (training data)
    train_X_bounds: Lower (train_X_bounds[0]) and Upper (train_X_bounds[1]) bounds
    N_eval_var_y: Number of Monte-Carlo simulations to estimate the total model variance
    N_text_X: Number of test data evaluated in batch
    N_Xi: Number of fixed inputs (inner loop)
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
        test_X = torch.from_numpy(sampling(N_test_X))
        prevC = 0  # Initialize the previous cost function
        pi_jdxplus = []  # Initialize the index to be fixed

        N_inputs = test_X.shape[-1]  # Number of inputs

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
                # Retlive the upper and the lower bounds of inputs
                X_upper_vec = torch.empty([1, len(minus_pi_jdxplus)])
                X_lower_vec = torch.empty_like(X_upper_vec)
                for hdx, hdx_val in enumerate(minus_pi_jdxplus):
                    X_lower_vec[:, hdx] = train_X_bounds[0, hdx_val]
                    X_upper_vec[:, hdx] = train_X_bounds[1, hdx_val]
                # Draw input values from the distributions
                Xi = (X_upper_vec - X_lower_vec) * torch.rand(
                    N_Xi, len(minus_pi_jdxplus)) + X_lower_vec
                # Initialize the variance
                var_pred_Xi_mean = torch.empty(N_Xi)

                for kdx in range(N_Xi):
                    test_Xi = test_X.detach().clone()
                    # Fix Xi
                    test_Xi[:, minus_pi_jdxplus] = Xi[kdx, :]
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        pred_Xi = gp.posterior(test_Xi)
                        pred_Xi_mean = pred_Xi.mean  # Prediction mean
                        if norm_flag is True:
                            # Normalize by the total variance
                            var_pred_Xi_mean[kdx] = torch.var(pred_Xi_mean) \
                                / var_pred_eval_X_mean
                        else:
                            var_pred_Xi_mean[kdx] = torch.var(pred_Xi_mean)

                # Take mean wrt. pi_jdx
                c_P_union_pi = torch.mean(var_pred_Xi_mean)
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

    return var_pred_eval_X_mean, shap / counter
