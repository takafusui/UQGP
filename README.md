# UQGP: Uncertainty Quantification based on Gaussian Process

This package performs uncertainty quantification (UQ) based on the Gaussian process (GP) based surrogated model.

## Getting Started
This project involves two sub-modules:

- `GP` folder contains Gaussian Process-related utilities based on `PyTorch`,
`GPyTorch`, and `BoTroch`.
- `UQ` folder contains Uncertainty Quantification-related functions to compute
the Sobol indices, the Shapley values, and the univariate effects.

## Prerequisites
In this project, you depend on `Tensorflow`, `PyTorch`, `GPyTorch`, `BoTorch`, 
`Hydra`, among others.
We highly recommend creating a `virtualenv` and installing necessary packages
within this environment.

Although `Tensorflow` and `PyTorch` would cause several dependency problems, we advise to install the following versions or higher:
```bash
tensorflow==2.3.0
torch==2.0.1
gpytorch==1.10
botorch==0.8.5
hydra-core==1.3.1
```

## Installation
Download from [here](https://github.com/takafusui/UQGP/) and install the package somewhere your `PYTHONPATH` is accessible.

## Directory structure
This package is assumed to be used on top of the policy functions and tensorflow data directory from the DEQN package.

The directory should be started from `USE_CONFIG_FROM_RUN` and should look like:
```bash
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
- `UQGPPreProcess.py` instantiates UQ analysis and stores common settings throughout the analysis. For instance, you can set the number of experimental design `N_train_X`, the model name `dir_name`, and the list of quantities of interest `QoIs_list`, among others.
- `gen_simulate_QoIs_train.py` is used to generate the initial experimental design. The generated samples are stored in `UQ/model_name/trainXYZ_LHS` directory.
    - `model_name` is defined in`UQGPPreProcess.py`. See the `dir_name` entity.
    - In `trainXYZ_LHS`, `XYZ` refers to the number of initial experimental design. You set `XYZ` in the `N_train_X` entity in `.hydra/config.yaml`. For testing purposes, you select `XYZ` to a small number (let's say 10), but for actual computations, `XYZ` should be big enough (let's say 200) depending on your applications.
- `sobol` directory includes the final figures (`pdf`) and the first-order Sobol indices in a `csv` format.
- `shapley` directory includes the final figures (`pdf`) and the Shapley values in a `csv` format.
- `univariate` directory includes the final figures (`pdf`) and the univariate effects in a `csv` format.
- `QoI.csv` is the outcome of the initial experimental design. You set the QoIs to be investigated in `UQGPPreProcess`. See `QoIs_list`.

## Usage
### Experimental design
To generate experimental design, execute:

```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python gen_simulate_QoIs_train.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```
Here you override `N_train_X` by setting `constants.constants.N_train_X=XYZ`.

The tentative results are stored in the `UQ` directory as a `csv` format. Once sampling is over, the results move to `UQ/model_name/trainXYZ_LHS` directory.

### First-order Sobol indices
To compute the first-order Sobol indices based on GP, execute:

```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python uq_gp_sobol.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```

The results are stored in `UQ/model_name/trainXYZ_LHS/sobol` as `csv` data.

To plot the first-order Sobol indices, execute:
```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python plt_sobol_gp.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```
The figures are stored in `UQ/model_name/trainXYZ_LHS/sobol/figs`.

### Shapley values
To compute the Shapley values based on GP, execute:

```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python uq_gp_shapley.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```

The results are stored in `UQ/model_name/trainXYZ_LHS/shapley` as `csv` data.

To plot the Shapley values, execute:
```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python plt_shapley_gp.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```
The figures are stored in `UQ/model_name/trainXYZ_LHS/shapley/figs`.

### Univariate effects
To compute the univariate effects based on GP, execute:

```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python uq_gp_univariate.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```

The results are stored in `UQ/model_name/trainXYZ_LHS/univariate` as `csv` data.

To plot the univariate effects, execute:
```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_deqn_data && python plt_univariate_gp.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.N_train_X=XYZ
```
The figures are stored in `UQ/model_name/trainXYZ_LHS/univariate/figs`.

## LOO error analysis
Under construction...

## Bayesian active learning
Under construction...

## Tail learning
In `post_process_learn.py`, we also compute
1. The prior distribution of ECS as like Roe and Baker (2007). Search around `l.551`.
2. The posterior distribution of ECS as like Kelly and Tan (2015). Search around `l. 578`.
3. The distribution of ECS implied by the mean of posterior as like Kelly and Tan (2015). Search around `l.604`.
4. Expected learning time to complete tail-learning as like Kelly and Tan (2015). Search around `l.650`.

To simulate the economy, we assume the true climate feedback parameter `truef`. Execute `post_process_learn.py` by adding your `truef` parameter such as `truef=0.65`
```bash
export USE_CONFIG_FROM_RUN_DIR=outputs/path_to_dean_data && python post_process_learn.py STARTING_POINT=LATEST hydra.run.dir=$USE_CONFIG_FROM_RUN_DIR constants.constants.truef=0.65
```

Note that all figures in `pdf` and tail learning time in `csv` are stored in your output directory (the same directory where you saved the distribution plots, etc.).