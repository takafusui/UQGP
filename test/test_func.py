#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: testfunc.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:
Test functions for uncertainty quantification
"""
import numpy as np
import torch

class TestFunc:
    """Correction of the test functions for uncertainty quantification."""

    def __init__(self, train_X):
        """Initialize TestFunc class."""
        self.train_X = train_X

    def g_func(self, train_X, marrel_or_goda):
        """Define the g-function of Sobol.

		n: Number of training datapoints
		d: Number of input parameters
		train_X.shape = (n, d)
		g_func.shape = (n, 1)
		"""
		# g-function
        _g_func = np.empty((train_X.shape[0], ))
        _g_func_idx = np.empty_like(train_X)

        for idx in range(train_X.shape[1]):
			# Set the coefficients
            if marrel_or_goda == 'marrel':
                aidx = idx + 1  # aidx = 1, 2,...
            elif marrel_or_goda == 'goda':
                aidx = idx

            _g_func_idx[:, idx] = (
				np.absolute(4 * train_X[:, idx] - 2) + aidx) / (1 + aidx)

		# Take the product of array elements
        _g_func = np.prod(_g_func_idx, axis=1)

		# Convert to torch from numpy
        _g_func = torch.from_numpy(_g_func)

        return _g_func[:, None]

    def ishigami_func(self, train_X):
        """Define the Ishigami function."""
        _ishigami = (
            torch.sin(train_X[:, 0])
            + 7 * torch.sin(train_X[:, 1])**2
            + 0.1 * train_X[:, 2]**4 * torch.sin(train_X[:, 0])
            )

        return _ishigami[:, None]