#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: gfunc_sobol_shapley.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
The first-order Sobol' indices and the Shapley values
"""
import numpy as np
import pandas as pd
from smt.sampling_methods import Random, LHS
import torch
from botorch.fit import fit_gpytorch_mll
import matplotlib.pyplot as plt
from matplotlib import rc

from UQGP.GP import GPutils
from UQGP.GP import ComputeErrorGP
from UQGP.UQ import SobolGP
from UQGP.UQ import ShapleyGP
from test_func import TestFunc

# --------------------------------------------------------------------------- #
# Plot setting
# --------------------------------------------------------------------------- #
# Use TeX font
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
# rc('text', usetex=True)

# Figure size
fsize = (9, 6)

# Font size
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.title_fontsize"] = 14

# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #
N_train_X = 256
col_header = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$']
N_inputs = len(col_header)
Xlimits = np.tile([0., 1.], (N_inputs, 1))
train_X_bounds = torch.from_numpy(
    np.tile(Xlimits[0].reshape(2, 1), (1, N_inputs)))

# Latin-Hypercube sampling
sampler = LHS(xlimits=Xlimits)
train_X = torch.from_numpy(sampler(N_train_X))

testfunc = TestFunc(train_X)
train_y = testfunc.g_func(train_X, 'marrel')

# --------------------------------------------------------------------------- #
# Predictivity coefficient Q2
# --------------------------------------------------------------------------- #
# Initialize a GP model
mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
# Train the GP model based on train_X and train_y
fit_gpytorch_mll(mll)  # Using BoTorch's fitting routine

# Test sample
N_test_X = 10000
sampler_LHS = LHS(xlimits=Xlimits)
test_X = torch.from_numpy(sampler_LHS(N_test_X))
test_y = testfunc.g_func(test_X, 'marrel')

Q2 = ComputeErrorGP.Q2_GP(gp, test_X, test_y)
print(r"Predictivity coefficient Q2: {:.3f}".format(Q2))

# --------------------------------------------------------------------------- #
# First-order Sobol' indices using prediction only
# We aim to replicate Marrel et al. (2009) Section 4.3
# --------------------------------------------------------------------------- #
""" N_Xi = 1000
S1st_pred = SobolGP.compute_S1st_pred(
    train_X, train_y, train_X_bounds, test_X, N_Xi)

print(r"First-order Sobol' indices are:")
print(S1st_pred)

# Plot the approximated and analytic first-order Sobol' indices
S1st_analytic = np.array([0.48, 0.21, 0.12, 0.08, 0.05])

# Plot
fig, ax = plt.subplots(figsize=fsize)
width = 0.45  # the width of the bars
x = np.arange(len(col_header))  # the label locations
ax.bar(x - width/2, S1st_pred, width, label='GP')
ax.bar(x + width/2, S1st_analytic, width, label='Analytic')
# Set the tick positions and labels
ax.set_xticks(range(len(col_header)))
ax.set_xticklabels(col_header)
ax.set_ylabel(r"$S_{1}$")
ax.legend()
plt.savefig('figs/gfunc_sobol.png', bbox_inches="tight")
plt.close()
 """
# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
# Shapley values
# We aim to replicate Gonda (2021) Fig. 3
# Note that there are 10 inputs. The exact method needs to traverse all 10!
# permuted set that is expensive. We use only the approx method here.
# --------------------------------------------------------------------------- #
N_train_X = 512
col_header = [
    r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$',
    r'$x_6$', r'$x_7$', r'$x_8$', r'$x_9$', r'$x_{10}$']
N_inputs = len(col_header)
Xlimits = np.tile([0., 1.], (N_inputs, 1))
train_X_bounds = torch.from_numpy(
    np.tile(Xlimits[0].reshape(2, 1), (1, N_inputs)))

# Latin-Hypercube sampling
sampler = LHS(xlimits=Xlimits)
# Test sample
N_test_X = 10000
sampler_LHS = LHS(xlimits=Xlimits)
test_X = torch.from_numpy(sampler_LHS(N_test_X))
test_y = testfunc.g_func(test_X, 'goda')

train_X = torch.from_numpy(sampler(N_train_X))
testfunc = TestFunc(train_X)
train_y = testfunc.g_func(train_X, 'goda')

# Initialize a GP model
mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
# Train the GP model based on train_X and train_y
fit_gpytorch_mll(mll)  # Using BoTorch's fitting routine

Q2 = ComputeErrorGP.Q2_GP(gp, test_X, test_y)
print(r"Predictivity coefficient Q2 is {:.3f}".format(Q2))

N_montecarlo = 100
var_y_approx = np.empty(N_montecarlo)
shap_approx = np.empty((N_montecarlo, N_inputs))

for idx in range(N_montecarlo):
	print(r"Iterattion: {} / {}".format(idx+1, N_montecarlo))
	var_y_approx[idx], shap_approx[idx, :] = ShapleyGP.compute_shapley_gp(
        train_X, train_y, train_X_bounds,
        N_eval_var_y=10000, N_inner=3, N_outer=1,
        exact_or_approx='approx', max_counter=3000, norm_flag=False)

var_y_approx_mean = np.mean(var_y_approx)
shap_approx_df = pd.DataFrame(shap_approx)
shap_approx_mean = shap_approx_df.mean(axis=0)
shap_approx_std = shap_approx_df.std(axis=0)

# Plot the Shapley values using the exact and approximation method
fig, ax = plt.subplots(figsize=fsize)
width = 0.75  # the width of the bars
x = np.arange(len(col_header))  # the label locations
ax.bar(-1, var_y_approx_mean, width, label='Overall variance')
ax.bar(x, shap_approx_mean, width, yerr=shap_approx_std, capsize=4, label='Approx')
# Set the tick positions and labels
ax.set_xticks(range(-1, len(col_header)))
ax.set_xticklabels([''] + col_header)
ax.set_ylabel(r"$Sh_{i}$")
ax.legend()
plt.savefig('figs/gfunc_shapley.png', bbox_inches="tight")
plt.close()

import ipdb; ipdb.set_trace()
