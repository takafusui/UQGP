#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
"""
Filename: GPutils.py
Author(s): Takafumi Usui
E-mail: u.takafumi@gmail.com
Description:

"""
import gpytorch
from botorch.models import SingleTaskGP
# from botorch.models.transforms.outcome import Standardize
# from botorch.models.transforms.input import InputStandardize
from botorch.models.transforms import Normalize, Standardize


def initialize_GP(train_X, train_y, train_X_bounds):
    """Initialize a GP model.

    trian_X.shape: [N, d]
    train_y.shape: [N, 1]

    Return the fitted GP model.
    """
    # Get the dimensions of train_X and train_y to standardize inputs
    train_X_dim = train_X.shape[-1]
    train_y_dim = train_y.shape[-1]

    gp = SingleTaskGP(
        train_X, train_y,
        input_transform=Normalize(d=train_X_dim, bounds=train_X_bounds),
        outcome_transform=Standardize(m=train_y_dim)
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    return mll, gp