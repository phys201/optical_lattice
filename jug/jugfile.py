from fidelity_sweep import run_sweep
from jug import TaskGenerator
import numpy as np



@TaskGenerator
def sweep_line(n_phot, stds, n_background, num_average):

    fidelities_averages_std = np.zeros((num_sweeps, 2))

    # Sweep one line
    for i, std in enumerate(stds):
        fidelities = np.zeros(num_average)
        for i, k in enumerate(range(num_average)):
            fidelities[k]=run_sweep(
                N_photon=int(n_phot[0]),
                std=std,
                N_back=n_background,
            )

        # store average fidelities std std
        fidelities_averages_std[i, 0] = np.average(fidelities)
        fidelities_averages_std[i, 1] = np.std(fidelities)

    row_dict = {
        n_phot[0]: fidelities_averages_std
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
def get_settings(stds, n_photons, num_sweeps, num_average):
    settings_dict = {
        'stds':        stds,
        'n_photons':    n_photons,
        'num_sweeps':   num_sweeps,
        'num_aerages':  num_average
    }
    return settings_dict


# How many sweep values per axis to evaulate (resulting plot will be num_sweeps * num_sweeps)
num_sweeps = 2

# How often to average on each point
num_average = 1


# Background photon counts
n_phot_background = 300


# Sweeping range

# Average Photons per atom
n_phot_start = 30
n_phot_stop = 180

# Std of PSF
std_start = 3
std_stop = 10

# Generate sweeping variables
n_photons = np.linspace(n_phot_start, n_phot_stop, num_sweeps)
stds = np.linspace(std_start, std_stop, num_sweeps)

# Run full sweep by trying to start one process per line
fullresults = join([sweep_line([n_phot], stds, n_phot_background, num_average) for n_phot in n_photons])

# Store experimental settings
settings = get_settings(stds, n_photons, num_sweeps, num_average)
