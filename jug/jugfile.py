from fidelity_sweep import run_sweep
from jug import TaskGenerator
import numpy as np
import time
from time import sleep
from random import randint



@TaskGenerator
def sweep_line(n_phot, stds, num_average, N, M, lam_back, vinnies_model, first_n_phot):

    n_phot = int(n_phot[0])

    if n_phot == first_n_phot:
        print("Adding random delay")
        sleep(randint(5,10))

    fidelities_averages_std = np.zeros((num_sweeps, 2))
    # Sweep one line
    for i, std in enumerate(stds):
        fidelities = np.zeros(num_average)
        for k in range(num_average):

            fidelities[k]=run_sweep(
                N_photon=n_phot,
                std=std,
                N=N,
                M=M,
                lam_back=lam_back,
                real_mixture=vinnies_model
            )

        # store average fidelities std std
        fidelities_averages_std[i, 0] = np.average(fidelities)
        fidelities_averages_std[i, 1] = np.std(fidelities)

    row_dict = {
        n_phot: fidelities_averages_std
    }

    return row_dict


# Join intermediate row dictionaries
@TaskGenerator
def join(row_dicts):
    final_dict = {}
    for row_dict in row_dicts:
        final_dict.update( row_dict )
    return final_dict


@TaskGenerator
def get_settings(stds, n_photons, num_sweeps, num_average, N, M, lam_back, real_mixture):
    settings_dict = {
        'stds':             stds,
        'n_photons':        n_photons,
        'num_sweeps':       num_sweeps,
        'num_average':      num_average,
        'N':                N,
        'M':                M,
        'lam_back':         lam_back,
        'real_mixture':     real_mixture
    }
    return settings_dict


# Average Photons per atom

# How many sweep values per axis to evaulate (resulting plot will be num_sweeps * num_sweeps)
num_sweeps = 10

# How often to average on each point
num_average = 3

# Which model to use
vinnies_model = True


# Settings held constant during sweep
N = 4
M = 10
lam_back = 2000

n_phot_start = 10
n_phot_stop = 170

# Std of PSF
std_start = 2
std_stop = 14

# Generate sweeping variables
n_photons = np.linspace(n_phot_start, n_phot_stop, num_sweeps)
stds = np.linspace(std_start, std_stop, num_sweeps)

# Run full sweep by trying to start one process per line
fullresults = join([sweep_line([n_phot], stds, num_average, N, M, lam_back, vinnies_model, first_n_phot=n_phot_start) for n_phot in n_photons])

# Store experimental settings
settings = get_settings(stds, n_photons, num_sweeps, num_average, N, M, lam_back, real_mixture=vinnies_model)
