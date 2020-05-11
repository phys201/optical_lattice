from jug import value, set_jugdir
import jugfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
set_jugdir('optical_lattice/jug/jugfile.jugdata')



def read_jug_results(results_dict, settings_dict):
    """Transform jug results into the usual 2d array of fidelities."""

    num_sweeps = settings_dict['num_sweeps']


    fidelities = np.zeros((num_sweeps, num_sweeps))
    fidelities_std = np.zeros_like(fidelities)

    for i, num_phot in enumerate(settings_dict['n_photons']):
        fidelities[i:] = results_dict[num_phot][0]
        fidelities_std[i:] = results_dict[num_phot][0]

    return fidelities, fidelities_std


# Demonstrate we can retrieve results from jug data
results = value(jugfile.fullresults)
settings = value(jugfile.settings)


fidelities, fidelities_std = read_jug_results(results, settings)


# Plot Fidelities
df = pd.DataFrame(fidelities, columns=settings['stds']/settings['M'] )
df = df.set_index(settings['n_photons']/settings['num_bakground'])

fig = plt.figure(figsize=(8, 6))
ax = sns.heatmap(df, vmin=0, vmax=100, cbar_kws={'label': 'Fidelity in %'})
ax.set(xlabel='PSF width in lattice sites', ylabel='average # photons of atom / average # photons background')

plt.savefig('av_fidelities.pdf')


# Plot standard deviation
df = pd.DataFrame(fidelities_std, columns=settings['stds']/settings['M'] )
df = df.set_index(settings['n_photons']/settings['num_bakground'])

fig = plt.figure(figsize=(8, 6))
ax = sns.heatmap(df, cbar_kws={'label': 'Fidelity std. deviation'})
ax.set(xlabel='PSF width in lattice sites', ylabel='average # photons of atom / average # photons background')

plt.savefig('av_fidelities_std.pdf')

