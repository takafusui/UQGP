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
from smt.sampling_methods import Random, LHS
import torch
from botorch.fit import fit_gpytorch_model
import matplotlib.pyplot as plt
from matplotlib import rc

from UQGP.GP import GPutils
from UQGP.GP import ComputeErrorGP
from UQGP.UQ import SobolGP
from test_func import TestFunc

# --------------------------------------------------------------------------- #
# Plot setting
# --------------------------------------------------------------------------- #
# Use TeX font
rc('font', **{'family': 'sans-serif', 'serif': ['Helvetica']})
rc('text', usetex=True)

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
N_train_X = 200
col_header = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$']
N_inputs = len(col_header)
Xlimits = np.tile([0., 1.], (N_inputs, 1))
train_X_bounds = torch.from_numpy(
    np.tile(Xlimits[0].reshape(2, 1), (1, N_inputs)))

# Latin-Hypercube sampling
sampler = LHS(xlimits=Xlimits)
train_X = torch.from_numpy(sampler(N_train_X))

TestFunc = TestFunc(train_X)
train_y = TestFunc.g_func(train_X)

# --------------------------------------------------------------------------- #
# Predictivity coefficient Q2
# --------------------------------------------------------------------------- #
# Initialize a GP model
mll, gp = GPutils.initialize_GP(train_X, train_y, train_X_bounds)
# Train the GP model based on train_X and train_y
fit_gpytorch_model(mll)  # Using BoTorch's fitting routine

# Test sample
N_test_X = 10000
sampler_LHS = LHS(xlimits=Xlimits)
test_X = torch.from_numpy(sampler_LHS(N_test_X))
test_y = TestFunc.g_func(test_X)

Q2 = ComputeErrorGP.Q2_GP(gp, test_X, test_y)
print(r"Predictivity coefficient Q2: {:.3f}".format(Q2))

# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
# First-order Sobol' indices using prediction only
# --------------------------------------------------------------------------- #
N_Xi = 500
S1st_pred = SobolGP.compute_S1st_pred(
    train_X, train_y, train_X_bounds, test_X, N_Xi)

print(r"First-order Sobol' indices are:")
print(S1st_pred)

# --------------------------------------------------------------------------- #
# Plot the approximated and analytic first-order Sobol' indices
# --------------------------------------------------------------------------- #
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
plt.savefig('figs/gfunc_sobol.pdf', bbox_inches="tight")
plt.close()