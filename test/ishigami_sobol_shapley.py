#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: ishigami_sobol_shapley.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
The first-order Sobol' indices and the Shapley values
"""
import numpy as np
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
col_header = [r'$x_1$', r'$x_2$', r'$x_3$']
N_inputs = len(col_header)
Xlimits = np.tile([-np.pi, np.pi], (N_inputs, 1))
train_X_bounds = torch.from_numpy(
	np.tile(Xlimits[0].reshape(2, 1), (1, N_inputs)))

# Latin-Hypercube sampling
sampler = LHS(xlimits=Xlimits)
train_X = torch.from_numpy(sampler(N_train_X))

testfunc = TestFunc(train_X)
train_y = testfunc.ishigami_func(train_X)

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
test_y = testfunc.ishigami_func(test_X)

Q2 = ComputeErrorGP.Q2_GP(gp, test_X, test_y)
print(r"Predictivity coefficient Q2: {:.3f}".format(Q2))

# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
print(r"First-order Sobol' indices using prediction only")
# --------------------------------------------------------------------------- #
N_Xi = 1000
var_y, S1st_pred = SobolGP.compute_S1st_pred(
	train_X, train_y, train_X_bounds, test_X, N_inner=N_Xi, norm_flag=True)

print(r"First-order Sobol' indices are:")
print(S1st_pred)

# --------------------------------------------------------------------------- #
# Plot the approximated and analytic first-order Sobol' indices
# --------------------------------------------------------------------------- #
S1st_analytic = np.array([0.3139, 0.4424, 0.])

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
plt.savefig('figs/ishigami_sobol.png', bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------------- #
print(r"Shapley values using the exact and approx methods")
# --------------------------------------------------------------------------- #
var_pred_eval_X_mean_exact, shap_exact = ShapleyGP.compute_shapley_gp(
	train_X, train_y, train_X_bounds,
	N_eval_var_y=10000, N_inner=3, N_outer=3000,
	exact_or_approx='exact', max_counter=None, norm_flag=False)

var_pred_eval_X_mean_approx, shap_approx = ShapleyGP.compute_shapley_gp(
	train_X, train_y, train_X_bounds,
	N_eval_var_y=10000, N_inner=3, N_outer=1,
	exact_or_approx='approx', max_counter=3000, norm_flag=False)

# Plot
fig, ax = plt.subplots(figsize=fsize)
width = 0.35  # the width of the bars
x = np.arange(len(col_header))  # the label locations
ax.bar(-1, var_pred_eval_X_mean_exact, width, label='Overall variance (Exact)')
ax.bar(x - width/2, shap_exact, width, label='Exact')
ax.bar(x + width/2, shap_approx, width, label='Approx')
# Set the tick positions and labels
ax.set_xticks(range(-1, len(col_header)))
ax.set_xticklabels([''] + col_header)
ax.set_ylabel(r"$Sh_{i}$")
ax.legend()
plt.savefig('figs/ishigami_shapley.png', bbox_inches="tight")
plt.close()