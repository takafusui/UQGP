#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: ishigami_activelearning.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Active learning to increase the global accuracy of GP
"""
import numpy as np
from smt.sampling_methods import Random, LHS
import torch
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

from UQGP.GP import ComputeErrorGP
from UQGP.GP import ActiveLearning
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
N_train_X_init = 200
col_header = [r'$x_1$', r'$x_2$', r'$x_3$']
N_inputs = len(col_header)
Xlimits = np.tile([-np.pi, np.pi], (N_inputs, 1))
train_X_bounds = torch.from_numpy(
	np.tile(Xlimits[0].reshape(2, 1), (1, N_inputs)))

# Latin-Hypercube sampling
sampler = LHS(xlimits=Xlimits)
train_X_init = torch.from_numpy(sampler(N_train_X_init))

testfunc = TestFunc()
train_y_init = testfunc.ishigami_func(train_X_init)

# Test sample
N_test_X = 10000
sampler_LHS = LHS(xlimits=Xlimits)
test_X = torch.from_numpy(sampler_LHS(N_test_X))
test_y = testfunc.ishigami_func(test_X)

# Botorch optimize_acqf parameters
# Number of initial conditions from which we optimize an acquisition function
# The larger NUM_RESTARTS, the more memory we need
N_restarts = 20  # Default number in Ax
# The larger RAW_SAMPLES, the better initial condition we have
raw_samples = 1024  # Default number in Ax is 1024
# Larger N would lead to more precise integration of the posterior variance,
# but since GP posterior variance is a relatively smooth function, would not
# expect pushing N larger to make a big difference.
N_MC_samples = 64  # Should be a power of 2
q_batch = 1  # != 1 when q>1 in optimize_acqf

verbose = 25

N_train_X_end = 250  # 400
N_train_X_list = np.arange(N_train_X_init, N_train_X_end + 1, verbose)

# --------------------------------------------------------------------------- #
# LHS
# --------------------------------------------------------------------------- #
""" errLOO = []
N_train_X_LOO = []

print(r"Using LHS")
for idx, N_train_X in enumerate(N_train_X_list):
	if idx == 0:
		# Use the same experimental design
		train_X = train_X_init
		train_y = train_y_init
	else:
		train_X = torch.from_numpy(sampler(N_train_X))
		train_y = testfunc.ishigami_func(train_X)
	err = ComputeErrorGP.leave_one_out_GP(train_X, train_y, train_X_bounds)
	errLOO.append(err[0])
	N_train_X_LOO.append(train_X.shape[0])

LHS_result = pd.DataFrame(np.vstack([N_train_X_LOO, errLOO]))
LHS_result.to_csv('csv/ishigami_LHS.csv', index=False, header=False) """

# --------------------------------------------------------------------------- #
# Bayesian active learning with the PV acquisition function
# --------------------------------------------------------------------------- #
""" errLOO = []
N_train_X_LOO = []

activelearning = ActiveLearning.BayesianActiveLearning(
	'PV', N_restarts, raw_samples, N_MC_samples, q_batch)
train_X, train_y = train_X_init, train_y_init
N_maxiter = N_train_X_list[-1] - N_train_X_list[0]

for idx in range(N_maxiter):

	if idx == 0:
		print(r"Enter the Bayesian active learning loop")
		err = ComputeErrorGP.leave_one_out_GP(train_X, train_y, train_X_bounds)
		errLOO.append(err[0])
		N_train_X_LOO.append(train_X.shape[0])

	train_X_add = activelearning.forward(train_X, train_y, train_X_bounds)
	train_y_add = testfunc.ishigami_func(train_X_add)

	# Add the candidate point
	train_X = torch.cat((train_X, train_X_add), axis=0)
	train_y = torch.cat((train_y, train_y_add), axis=0)


	if (idx + 1) % verbose == 0:
		print(r"Iteration: {} / {}".format(idx+ 1 , N_maxiter))
		err = ComputeErrorGP.leave_one_out_GP(train_X, train_y, train_X_bounds)
		errLOO.append(err[0])
		N_train_X_LOO.append(train_X.shape[0])

PV_result = pd.DataFrame(np.vstack([N_train_X_LOO, errLOO]))
PV_result.to_csv('csv/ishigami_PV.csv', index=False, header=False) """

# --------------------------------------------------------------------------- #
# Bayesian active learning with the qNIPV acquisition function
# --------------------------------------------------------------------------- #
errLOO = []
N_train_X_LOO = []

activelearning = ActiveLearning.BayesianActiveLearning(
	'qNIPV', N_restarts, raw_samples, N_MC_samples, q_batch)
train_X, train_y = train_X_init, train_y_init
N_maxiter = N_train_X_list[-1] - N_train_X_list[0]

for idx in range(N_maxiter):

	if idx == 0:
		print(r"Enter the Bayesian active learning loop")
		err = ComputeErrorGP.leave_one_out_GP(train_X, train_y, train_X_bounds)
		errLOO.append(err[0])
		N_train_X_LOO.append(train_X.shape[0])

	train_X_add = activelearning.forward(train_X, train_y, train_X_bounds)
	train_y_add = testfunc.ishigami_func(train_X_add)

	# Add the candidate point
	train_X = torch.cat((train_X, train_X_add), axis=0)
	train_y = torch.cat((train_y, train_y_add), axis=0)


	if (idx + 1) % verbose == 0:
		print(r"Iteration: {} / {}".format(idx+ 1 , N_maxiter))
		err = ComputeErrorGP.leave_one_out_GP(train_X, train_y, train_X_bounds)
		errLOO.append(err[0])
		N_train_X_LOO.append(train_X.shape[0])

NIPV_result = pd.DataFrame(np.vstack([N_train_X_LOO, errLOO]))
NIPV_result.to_csv('csv/ishigami_NIPV.csv', index=False, header=False)
# --------------------------------------------------------------------------- #
# Plot the LOO error
# --------------------------------------------------------------------------- #
LHS_read = pd.read_csv('csv/ishigami_LHS.csv', header=None)
PV_read = pd.read_csv('csv/ishigami_PV.csv', header=None)
NIPV_read = pd.read_csv('csv/ishigami_NIPV.csv', header=None)

# Plot
fig, ax = plt.subplots(figsize=fsize)
ax.plot(LHS_read.loc[0], LHS_read.loc[1], label='LHS')
ax.plot(PV_read.loc[0], PV_read.loc[1], label='PV')
ax.plot(NIPV_read.loc[0], NIPV_read.loc[1], label='NIPV')
ax.set_xlabel(r'Number of Experimental design')
ax.set_ylabel(r'$err_{\mathrm{LOO}}$')
ax.set_yscale('log')
ax.legend()
plt.savefig('figs/ishigami_activelearning.png', bbox_inches="tight")
plt.close()

import ipdb; ipdb.set_trace()