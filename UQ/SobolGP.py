#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: SobolGP.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Compute Sobol indices using a GP surrogate model
Reference:
Marrel et al. (2009)
"""
import numpy as np
import itertools
import torch
import gpytorch
# BoTorch
from botorch.fit import fit_gpytorch_mll

from UQGP.GP import GPutils


def compute_S1st_pred(train_X, train_y, train_X_bounds, test_X, N_Xi):
    """Compute the first-order Sobol' indices using the GP predictor only.

    n: Number of experimental design
    m: Number of test points
    train_X.shape = [n, N_inputs]
    train_y.shape = [n, 1]
    test_X.shape = [m, N_inputs]

    train_X, train_y: Experimental design points (training data)
    train_X_bounds: Lower (train_X_bounds[0]) and Upper (train_X_bounds[1]) bounds

    return
    S1st_pred.shape = [N_inputs]
    """
    # Get the number of inputs
    N_inputs = train_X.shape[-1]
    # Number of test points
    N_test_X = test_X.shape[0]

    # Initialize a GP model
    mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
    # Train the GP model based on train_X and train_y
    fit_gpytorch_mll(mll)  # Using BoTorch's routine

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # As we standardized inputs, we must use gp.posterior to get rescaled
        # outputs.
        pred_test_X = gp.posterior(test_X)
        pred_test_X_mean = pred_test_X.mean
        # Take the variance of the prediction mean over test_X (X1, ..., Xd)
        # It is a denominator in S1st_pred
        var_E_pred_test_X = torch.var(pred_test_X_mean)

    var_E_pred_test_Xi_mean = torch.empty(N_inputs)  # Numerator in S1st_pred

    # Computing the first-order Sobol' indices with the mean of prediction
    for idx in range(N_inputs):
        # Fix the model parameter i that is uniformly distributed
        Xi_sampler = torch.distributions.uniform.Uniform(
            train_X_bounds[0, idx], train_X_bounds[1, idx])
        Xi = Xi_sampler.sample((N_Xi, ))

        # Store the mean of the mean of predictions
        pred_test_Xi_mean = torch.empty([N_test_X, N_Xi])

        for jdx in range(N_Xi):
            # Fix parameter idx-th at Xi[jdx]
            test_Xi = test_X.detach().clone()
            test_Xi[:, idx] = Xi[jdx]

            # Simulate with Xi[jdx] fixed test_X data
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                # As we standardized inputs, we must use gp.posterior to get
                # rescaled outputs.
                pred_Xi = gp.posterior(test_Xi)
                pred_Xi_mean = pred_Xi.mean  # Prediction mean
            pred_test_Xi_mean[:, jdx] = pred_Xi_mean[:, 0]

        # Take the mean over X1, X2, ..., Xd (Xi.shape=N_test_X)
        E_pred_test_Xi_mean = torch.mean(pred_test_Xi_mean, axis=0)
        # Take the variance wrt. Xi (Xi.shape=N_Xi)
        var_E_pred_test_Xi_mean[idx] = torch.var(E_pred_test_Xi_mean)

    # Compute the first-order Sobol indices using the predictor-only
    S1st_pred = var_E_pred_test_Xi_mean / var_E_pred_test_X

    return S1st_pred


def compute_S2nd_pred(train_X, train_y, test_X, X_lower, X_upper, N_Xi, X_range):
    """Compute the second-order Sobol' indices using the predictor only.

    train_X.shape = [n, N_inputs]
    train_y.shape = [n, 1]
    n: Number of experimental design

    return
    S2nd_pred.shape = [N_inputs, N_inputs]
    We need to remove diagonal elemets as we are not interested in S2nd_pred_ij
    """
    # Get the dimensions of train_X and train_y
    # train_X_dim = train_X.shape[1]
    # train_y_dim = train_y.shape[1]

    # Get the number of inputs
    N_inputs = train_X.shape[-1]
    # Number of test points
    N_test_X = test_X.shape[0]

    # # Instantiate and train the GP model
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # gp = SingleTaskGP(
    #     train_X, train_y, likelihood=likelihood,
    #     input_transform=InputStandardize(d=train_X_dim),
    #     outcome_transform=Standardize(m=train_y_dim)
    # )
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    xlimits = np.stack((X_lower, X_upper), axis=1)
    train_X_bounds = torch.from_numpy(xlimits.T)
    mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
    # Train the GP model based on train_X and train_y
    fit_gpytorch_mll(mll)  # Using BoTorch's routine
    # Get into evaluation (predictive posterior) mode
    # gp.eval()
    # likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Total model output variance
        pred_test_X = gp.posterior(test_X)
        pred_test_X_mean = pred_test_X.mean
        var_E_pred_test_X = torch.var(pred_test_X_mean)

        # import ipdb; ipdb.set_trace()

    S2nd_pred = torch.empty(N_inputs, N_inputs)

    for idx in itertools.permutations(range(N_inputs), 2):
        # Generate the indices of [Xi, Xj] pair from uniform distributions
        Xi_sampler = torch.distributions.uniform.Uniform(
            X_range[0, idx[0]], X_range[1, idx[0]])
        Xj_sampler = torch.distributions.uniform.Uniform(
            X_range[0, idx[1]], X_range[1, idx[1]])
        # Draw N_Xi samples
        Xi = Xi_sampler.sample((N_Xi, ))
        Xj = Xj_sampler.sample((N_Xi, ))

        pred_test_Xi_mean = torch.empty([N_test_X, N_Xi])
        pred_test_Xj_mean = torch.empty([N_test_X, N_Xi])
        pred_test_Xij_mean = torch.empty([N_test_X, N_Xi])
        for jdx in range(N_Xi):
            test_Xi = test_X.detach().clone()
            test_Xj = test_X.detach().clone()
            test_Xij = test_X.detach().clone()
            test_Xi[:, idx[0]] = Xi[jdx]  # Fix the ith inputs
            test_Xj[:, idx[1]] = Xj[jdx]  # Fix the jth inputs
            test_Xij[:, idx[0]] = Xi[jdx]  # Fix the ith inputs
            test_Xij[:, idx[1]] = Xj[jdx]  # Fix the jth inputs

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_Xi = gp.posterior(test_Xi)
                pred_Xi_mean = pred_Xi.mean
                pred_Xj = gp.posterior(test_Xj)
                pred_Xj_mean = pred_Xj.mean
                pred_Xij = gp.posterior(test_Xij)
                pred_Xij_mean = pred_Xij.mean
            pred_test_Xi_mean[:, jdx] = pred_Xi_mean.flatten()
            pred_test_Xj_mean[:, jdx] = pred_Xj_mean.flatten()
            pred_test_Xij_mean[:, jdx] = pred_Xij_mean.flatten()

        # Take the mean wrt. Xi
        E_pred_test_Xi_mean = torch.mean(pred_test_Xi_mean, axis=0)
        E_pred_test_Xj_mean = torch.mean(pred_test_Xj_mean, axis=0)
        E_pred_test_Xij_mean = torch.mean(pred_test_Xij_mean, axis=0)
        # Take the variance wrt. Xi or Xj or Xij
        var_E_pred_test_Xi_mean = torch.var(E_pred_test_Xi_mean)
        var_E_pred_test_Xj_mean = torch.var(E_pred_test_Xj_mean)
        var_E_pred_test_Xij_mean = torch.var(E_pred_test_Xij_mean)

        # Compute the second-order Sobol' indices
        S2nd_pred[idx[0], idx[1]] = (
            var_E_pred_test_Xij_mean - var_E_pred_test_Xi_mean
            - var_E_pred_test_Xj_mean) / var_E_pred_test_X

    # Replace diagonal elements with 0.
    S2nd_pred[range(N_inputs), range(N_inputs)] = 0.

    return S2nd_pred


def compute_S1st_tilde(
        train_X, train_y, train_X_bounds, test_X, N_Xi, X_range, N_sampling):
    """Compute the first-order Sobol' indices using the global GP model.

    n: Number of experimental design
    m: Number of test points
    train_X.shape = [n, N_inputs]
    train_y.shape = [n, 1]
    test_X.shape = [m, N_inputs]

    train_X, train_y: Experimental design points (training data)
    train_X_bounds: Lower (train_X_bounds[0]) and Upper (train_X_bounds[1]) bounds

    return
    S1st_tilde.shape = [N_inputs]
    """
    # Get the dimensions of train_X and train_y
    # train_X_dim = train_X.shape[-1]
    # train_y_dim = train_y.shape[-1]

    # Get the number of inputs
    N_inputs = train_X.shape[-1]
    # Number of test points
    N_test_X = test_X.shape[0]

    # Train the GP model with the standardized inputs
    # likelihood = gpytorch.likelihoods.GaussianLikelihood(
    #     batch_shape=torch.Size([N_batch_size]))
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

    # Draw N_sampling samples from the trained GP model
    # Use torch.distributions.Normal() method
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # As we standardized inputs, we must use gp.posterior to get rescaled
        # outputs.
        pred_test_X = gp.posterior(test_X)
        pred_test_X_mean = pred_test_X.mean
        pred_test_X_variance = pred_test_X.variance
        Ytilde = torch.distributions.Normal(
            pred_test_X_mean[:, 0],
            torch.sqrt(pred_test_X_variance[:, 0])
        ).sample(sample_shape=torch.Size([N_sampling]))

    # Take the variance over X1, X2, ..., Xd (Xi.shape=N_test_X)
    var_Ytilde = torch.var(Ytilde, axis=1)
    # Take the mean over Omega (Omega.shape=N_sampling)
    E_var_Ytilde = torch.mean(var_Ytilde)  # Denominator

    Ytilde_Xi = torch.empty([N_sampling, N_test_X, N_Xi])
    var_E_Ytilde_Xi = torch.empty([N_inputs, N_sampling])

    for idx in range(N_inputs):
        # Fix the model parameter i that is assumed to be uniformly distributed
        Xi_sampler = torch.distributions.uniform.Uniform(
            X_range[0, idx], X_range[1, idx])
        Xi = Xi_sampler.sample((N_Xi, ))

        # E_Ytilde_Xi = torch.empty(N_Xi, N_sampling)

        for jdx in range(N_Xi):
            # Fix parameter idx at Xi[jdx]
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
        E_Ytilde_Xi = torch.mean(Ytilde_Xi, axis=1)
        # Take the variance over Xi (Xi.shape=N_Xi)
        var_E_Ytilde_Xi[idx, :] = torch.var(E_Ytilde_Xi, axis=1)

    # Take the mean over Omega (Omega.shape=N_sampling)
    mu_S1st_tilde = torch.mean(var_E_Ytilde_Xi, axis=1) / E_var_Ytilde
    # Take the variance over Omega(Omega.shape=N_sampling)
    sigma_S1st_tilde = torch.sqrt(
        torch.var(var_E_Ytilde_Xi, axis=1) / E_var_Ytilde**2)

    return mu_S1st_tilde, sigma_S1st_tilde


def compute_S2nd_tilde(train_X, train_y, test_X, X_lower, X_upper, N_Xi, X_range, N_sampling):
    """Compute the second-order Sobol' indices using the global GP model.

    train_X.shape = [n, N_inputs]
    train_y.shape = [n, 1]
    n: Number of experimental design

    return
    S2nd_tilde.shape = [N_inputs, N_inputs]
    We need to remove diagonal elemets as we are not interested in S2nd_pred_ij
    """
    # Get the dimensions of train_X and train_y
    # train_X_dim = train_X.shape[1]
    # train_y_dim = train_y.shape[1]

    # Get the number of inputs
    N_inputs = train_X.shape[-1]
    # Number of test points
    N_test_X = test_X.shape[0]

    # # Instantiate and train the GP model
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # gp = SingleTaskGP(
    #     train_X, train_y, likelihood=likelihood,
    #     input_transform=InputStandardize(d=train_X_dim),
    #     outcome_transform=Standardize(m=train_y_dim))
    # mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    xlimits = np.stack((X_lower, X_upper), axis=1)
    train_X_bounds = torch.from_numpy(xlimits.T)
    mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
    # Train the GP model based on train_X and train_y
    fit_gpytorch_mll(mll)  # Using BoTorch's routine
    # Get into evaluation (predictive posterior) mode
    # gp.eval()
    # likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Sampling N_sampling times from posterior
        pred_test_X = gp.posterior(test_X)
        pred_test_X_mean = pred_test_X.mean
        pred_test_X_variance = pred_test_X.variance
        Ytilde = torch.distributions.Normal(
            pred_test_X_mean.flatten(),
            torch.sqrt(pred_test_X_variance.flatten())).sample(
                sample_shape=torch.Size([N_sampling, ]))

    var_Ytilde = torch.var(Ytilde, axis=1)
    E_var_Ytilde = torch.mean(var_Ytilde)

    Y_tilde_Xi = torch.empty([N_sampling, N_test_X, N_Xi])
    Y_tilde_Xj = torch.empty_like(Y_tilde_Xi)
    Y_tilde_Xij = torch.empty_like(Y_tilde_Xi)

    S2nd_tilde = torch.empty(N_inputs, N_inputs, N_sampling)
    mu_S2nd_tilde = torch.empty(N_inputs, N_inputs)
    sigma_S2nd_tilde = torch.empty_like(mu_S2nd_tilde)

    for idx in itertools.permutations(range(N_inputs), 2):
        # Generate the indices of [Xi, Xj] pair from uniform distributions
        Xi_sampler = torch.distributions.uniform.Uniform(
            X_range[0, idx[0]], X_range[1, idx[0]])
        Xj_sampler = torch.distributions.uniform.Uniform(
            X_range[0, idx[1]], X_range[1, idx[1]])
        # Draw N_Xi samples
        Xi = Xi_sampler.sample((N_Xi, ))
        Xj = Xj_sampler.sample((N_Xi, ))

        E_Ytilde_Xi = torch.empty(N_Xi)
        for jdx in range(N_Xi):
            test_Xi = test_X.detach().clone()
            test_Xj = test_X.detach().clone()
            test_Xij = test_X.detach().clone()
            test_Xi[:, idx[0]] = Xi[jdx]  # Fix the ith inputs
            test_Xj[:, idx[1]] = Xj[jdx]  # Fix the jth inputs
            test_Xij[:, idx[0]] = Xi[jdx]  # Fix the ith inputs
            test_Xij[:, idx[1]] = Xj[jdx]  # Fix the jth inputs

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred_test_Xi = gp.posterior(test_Xi)
                Y_tilde_Xi[:, :, jdx] = torch.distributions.Normal(
                    pred_test_Xi.mean.flatten(),
                    torch.sqrt(pred_test_Xi.variance.flatten())).sample(
                        sample_shape=torch.Size([N_sampling, ]))

                pred_test_Xj = gp.posterior(test_Xj)
                Y_tilde_Xj[:, :, jdx] = torch.distributions.Normal(
                    pred_test_Xj.mean.flatten(),
                    torch.sqrt(pred_test_Xj.variance.flatten())).sample(
                        sample_shape=torch.Size([N_sampling, ]))

                pred_test_Xij = gp.posterior(test_Xij)
                Y_tilde_Xij[:, :, jdx] = torch.distributions.Normal(
                    pred_test_Xij.mean.flatten(),
                    torch.sqrt(pred_test_Xij.variance.flatten())).sample(
                        sample_shape=torch.Size([N_sampling, ]))

        # Take the mean wrt. the test_X
        E_Ytilde_Xi = torch.mean(Y_tilde_Xi, axis=1)
        E_Ytilde_Xj = torch.mean(Y_tilde_Xj, axis=1)
        E_Ytilde_Xij = torch.mean(Y_tilde_Xij, axis=1)
        # Take the variance wrt. Xi
        var_E_Ytilde_Xi = torch.var(E_Ytilde_Xi, axis=1)
        var_E_Ytilde_Xj = torch.var(E_Ytilde_Xj, axis=1)
        var_E_Ytilde_Xij = torch.var(E_Ytilde_Xij, axis=1)

        # Compute the second-order Sobol' indices
        S2nd_tilde[idx[0], idx[1]] = (
            var_E_Ytilde_Xij - var_E_Ytilde_Xi - var_E_Ytilde_Xj
        ) / E_var_Ytilde

        mu_S2nd_tilde[idx[0], idx[1]] = torch.mean(
            S2nd_tilde[idx[0], idx[1]])
        sigma_S2nd_tilde[idx[0], idx[1]] = torch.std(
            S2nd_tilde[idx[0], idx[1]])    # Replace diagonal elements with 0
    mu_S2nd_tilde[range(N_inputs), range(N_inputs)] = 0.
    sigma_S2nd_tilde[range(N_inputs), range(N_inputs)] = 0.
    return mu_S2nd_tilde, sigma_S2nd_tilde
