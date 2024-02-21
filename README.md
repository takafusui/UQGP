# UQGP: Uncertainty Quantification based on Gaussian Process

## Getting Started
This project involes two sub-modules:

- `GP` folder containts Gaussian Process related utilities based on `PyTorch`,
`GPyTorch` and `BoTroch`.
- `UQ` folder containts Uncertainty Quantification related functions to compute
the Sobol indices, the Shapley values and the univariate effects.

## Prerequisites
In this project, you depend on `Tensorflow`, `PyTorch`, `GPyTorch` and `BoTorch`.
I highly recommend to prepare a `virtualenv` and to install necessary packages
within this environment.

## Installation
Download from [here](https://github.com/takafusui/UQGP/) and install the package
somewhere your `PYTHONPATH` is accessible.

## Directory
This package is assumed to be used on top of the policy functions and tensorflow data
directory from the DEQN package.

Make a directory where you would contain your UQ results. The directory should be:
```
USE_CONFIG_FROM_RUN
├── UQGPPreProcess.py
├── gen_simulate_QoIs_train.py
├── uq_gp_sobol.py
├── uq_gp_shapley.py
├── uq_gp_univariate.py
├── plt_sobol_gp.py
├── plt_shapley_gp.py
├── plt_univariate_gp.py
│
└── UQ/model_name/trainXYZ_LHS
    │
    ├── sobol
    │   ├──figs
    │   └──S1st_pred_QoI.csv 
    │
    ├── shapley
    │   ├──figs
    │   ├──shapley_exact_QoI.csv
    │   └──shapley_approx_QoI.csv 
    │
    ├── univariate
    │   ├──figs
    │   ├──train_X_bounds.csv
    │   └──uni_pred_QoI.csv 
    │
    └── QoI.csv
```
- `USE_CONFIG_FROM_RUN` comes from DEQN library
- `gen_simulate_QoIs_train.py` is used to generate the initial experimental design.
The generated samples is stored in `UQ/model_name/trainXYZ_LHS` directory.
    - `model_name` is defined in`UQGPPreProcess.py`. See `dir_name`.
    - In `trainXYZ_LHS`, `XYZ` refers to the number of initial experimental design.
    You set `XYZ` in the `N_train_X` entity in `.hydra/config.yaml`.
- `sobol` directory containts the final figures as well as the first-order Sobol indices in `.csv`.
- `shapley` directory containts the final figures as well as the Shapley values in `.csv`.
- `univariate` directory containts the final figures as well as the univariate effects in `.csv`.
- `QoI.csv` is the results of the initial experimental design. You set the QoIs to be investigated 
in `UQGPPreProcess`. See `QoIs_list`.

## Usage
### Experimental design
To generate experimental design, execute something like
```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python gen_simulate_QoIs_train.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```
Here you can override `N_train_X` by setting `constants.constants.N_train_X=XYZ`.

The tentative results are stored in `UQ` directory as a CSV format. Once sampling is over,
the results move to `UQ/model_name/trainXYZ_LHS` directory.

### First-order Sobol indices
To compute the first-order Sobol indices based on GP, execute something like:

```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python uq_gp_sobol.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```

The results are stored in `UQ/model_name/trainXYZ_LHS/sobol` as CSV data.

To plot the first-order Sobol indices, execute:
```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python plt_sobol_gp.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```
The figures are stored in `UQ/model_name/trainXYZ_LHS/sobol/figs`.

### Shapley values
To compute the Shapley values based on GP, execute something like:

```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python uq_gp_shapley.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```

The results are stored in `UQ/model_name/trainXYZ_LHS/shapley` as CSV data.

To plot the first-order Sobol indices, execute:
```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python plt_shapley_gp.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```
The figures are stored in `UQ/model_name/trainXYZ_LHS/shapley/figs`.

### Univariate effects

## Bayesian active learning
Under construction...
