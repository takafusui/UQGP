#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: hartmanni6_activelearning.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Active learning to increase the global accuracy of GP
We use the six-dimensional Hartmann function as a test function.
"""
import numpy as np
from smt.sampling_methods import Random, LHS
import torch
from botorch.test_functions.synthetic import Hartmann
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

from UQGP.GP import ComputeErrorGP
from UQGP.GP import ActiveLearning

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
N_train_X_init = 100  # Initial experimental design size
# Six-dimensional test function
col_header = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$', r'$x_6$']
N_inputs = len(col_header)
# Evaluated on [0, 1]^N_inputs
Xlimits = np.tile([0., 1.], (N_inputs, 1))
# Hartmann test function
hartmann6 = Hartmann(dim=N_inputs)
train_X_bounds = torch.from_numpy(
	np.tile(Xlimits[0].reshape(2, 1), (1, N_inputs)))

# Latin-Hypercube sampling and the initial experimental design
sampler = LHS(xlimits=Xlimits)
train_X_init = torch.from_numpy(sampler(N_train_X_init))
train_y_init = hartmann6.evaluate_true(train_X_init)[:, None]

# --------------------------------------------------------------------------- #
# Botorch and optimize_acqf parameters
# --------------------------------------------------------------------------- #
# Number of initial conditions from which we optimize an acquisition function
# The larger NUM_RESTARTS, the more memory we need
N_restarts = 10  # Default number in Ax is 20
# The larger RAW_SAMPLES, the better initial condition we have
raw_samples = 1024  # Default number in Ax
# Larger N would lead to more precise integration of the posterior variance,
# but since GP posterior variance is a relatively smooth function, would not
# expect pushing N larger to make a big difference.
N_MC_samples = 256  # Should be a power of 2
q_batch = 1  # when q > 1 in optimize_acqf, q =! 1

# Error analysis setting
n_mc_iter = 100  # Number of Monte-Carlo iteration
verbose = 20
N_train_X_end = 200
N_train_X_list = np.arange(N_train_X_init, N_train_X_end + 1, verbose)

# --------------------------------------------------------------------------- #
# Using LHS
# --------------------------------------------------------------------------- #
errLOO = np.empty((n_mc_iter, len(N_train_X_list)))
N_train_X_LOO = []

print(r"Space filling design using LHS")

for jdx in range(n_mc_iter):
	print(r"Monte-Carlo iteration {} / {}".format(jdx + 1, n_mc_iter))

	for idx, N_train_X in enumerate(N_train_X_list):
		if idx == 0:
			# Use the same initial experimental design
			train_X = train_X_init
			train_y = train_y_init
		else:
			train_X = torch.from_numpy(sampler(N_train_X))
			train_y = hartmann6.evaluate_true(train_X)[:, None]
		# Compute a LOO error
		err = ComputeErrorGP.leave_one_out_GP(train_X, train_y, train_X_bounds)
		errLOO[jdx, idx] = err[0]

		if jdx == 0:
			# As N_train_X_LOO is the same for all jdx
			N_train_X_LOO.append(train_X.shape[0])

LHS_result = pd.DataFrame(np.vstack([N_train_X_LOO, errLOO]))
LHS_result.to_csv('csv/hartmann6_LHS.csv', index=False, header=False)

# --------------------------------------------------------------------------- #
# Bayesian active learning
# --------------------------------------------------------------------------- #
def MCBaisianActiveLearning(
		criterion, n_mc_iter, n_max_iter, train_X_init, train_y_init, verbose):
	"""Monte-Carlo Bayesian active learning.
	criterion: Acquisition function
	n_mc_iter: Number of Monte-Carlo iteration
	n_max_iter: Number of max Bayesian active learning iteration
	train_X_init: Initial training input
	train_y_init: Initial training output
	verbose: Print verbose
	"""
	# Instantiate active learning object
	activelearning = ActiveLearning.BayesianActiveLearning(
		criterion, N_restarts, raw_samples, N_MC_samples, q_batch)

	# Initialize output arrays
	errLOO = np.empty((n_mc_iter, n_max_iter//verbose + 1))
	N_train_X_LOO = []


	print(r"Bayesian active learning using the {} criterion".format(criterion))

	for jdx in range(n_mc_iter):
		print(r"Monte-Carlo iteration {} / {}".format(jdx + 1, n_mc_iter))
		# Initialization
		train_X, train_y = train_X_init, train_y_init

		for idx in range(n_max_iter):

			if idx == 0:
				err = ComputeErrorGP.leave_one_out_GP(
					train_X, train_y, train_X_bounds)
				errLOO[jdx, idx] = err[0]
				if jdx == 0:
					N_train_X_LOO.append(train_X.shape[0])

			train_X_add = activelearning.forward(train_X, train_y, train_X_bounds)
			train_y_add = hartmann6.evaluate_true(train_X_add)[:, None]

			# Add the candidate point
			train_X = torch.cat((train_X, train_X_add), axis=0)
			train_y = torch.cat((train_y, train_y_add), axis=0)


			if (idx + 1) % verbose == 0:
				print(r"Iteration: {} / {}".format(idx+ 1 , n_max_iter))
				err = ComputeErrorGP.leave_one_out_GP(train_X, train_y, train_X_bounds)
				errLOO[jdx, (idx + 1) // verbose] = err[0]
				if jdx == 0:
					N_train_X_LOO.append(train_X.shape[0])
	return N_train_X_LOO, errLOO

# --------------------------------------------------------------------------- #
# Using the PSTD acquisition function
# --------------------------------------------------------------------------- #
n_max_iter = N_train_X_list[-1] - N_train_X_list[0]
N_train_X_LOO, errLOO = MCBaisianActiveLearning(
	'PSTD', n_mc_iter, n_max_iter, train_X_init, train_y_init, verbose)

PSTD_result = pd.DataFrame(np.vstack([N_train_X_LOO, errLOO]))
PSTD_result.to_csv('csv/hartmann6_PSTD.csv', index=False, header=False)

# --------------------------------------------------------------------------- #
# Using the qNIPV acquisition function
# --------------------------------------------------------------------------- #
N_train_X_LOO, errLOO = MCBaisianActiveLearning(
	'qNIPV', n_mc_iter, n_max_iter, train_X_init, train_y_init, verbose)

NIPV_result = pd.DataFrame(np.vstack([N_train_X_LOO, errLOO]))
NIPV_result.to_csv('csv/hartmann6_NIPV.csv', index=False, header=False)

# import ipdb; ipdb.set_trace()
# --------------------------------------------------------------------------- #
# Plot the LOO error
# --------------------------------------------------------------------------- #
# Read the LOO error history
LHS_read = pd.read_csv('csv/hartmann6_LHS.csv', header=None)
PSTD_read = pd.read_csv('csv/hartmann6_PSTD.csv', header=None)
NIPV_read = pd.read_csv('csv/hartmann6_NIPV.csv', header=None)

LHS_mu, LHS_std = LHS_read.loc[1:].mean(), LHS_read.loc[1:].std()
PSTD_mu, PSTD_std = PSTD_read.loc[1:].mean(), PSTD_read.loc[1:].std()
NIPV_mu, NIPV_std = NIPV_read.loc[1:].mean(), NIPV_read.loc[1:].std()

# Plot
fill_alpha = 0.3
fig, ax = plt.subplots(figsize=fsize)
ax.plot(LHS_read.loc[0], LHS_mu, label='LHS')
ax.fill_between(LHS_read.loc[0], LHS_mu - LHS_std, LHS_mu + LHS_std, alpha=fill_alpha)
ax.plot(PSTD_read.loc[0], PSTD_mu, label='PSTD')
ax.fill_between(PSTD_read.loc[0], PSTD_mu - PSTD_std, PSTD_mu + PSTD_std, alpha=fill_alpha)
ax.plot(NIPV_read.loc[0], NIPV_mu, label='NIPV')
ax.fill_between(NIPV_read.loc[0], NIPV_mu - NIPV_std, NIPV_mu + NIPV_std, alpha=fill_alpha)
ax.set_xlabel(r'Number of experimental design')
ax.set_ylabel(r'$err_{\mathrm{LOO}}$')
ax.set_yscale('log')
ax.legend()
plt.savefig('figs/hartmann6_activelearning.png', bbox_inches="tight")
plt.close()
