import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from itertools import product
import seaborn as sns


def photon_counts(N, M, N_atom, N_photon, std=1, N_backg=100, lam_backg=1, plot=True):
    '''
    Generates positions of photon counts from the randomly placed atoms on a lattice and from Poissonian dark counts.

    Parameters
    ----------
    N : integer
        number of lattice sites along one direction (NxN)
    M: integer
        number of camera pixels per lattice site along one direction (MxM)
    N_atom: integer
        total number of atoms on the lattice
    std: float
        standard deviation of the Gaussian that is sampled from
    N_photons: integer
        number of photons sampled from an atom
    N_backg: integer
        number of samples drawn from the Poisson distribution for the background noise
    lam_back: float
        expectation interval of the Poisson dark count event

    Returns
    -------
    xloc, yloc = array
        x and y positions of all the photon counts
    '''

    #Randomly place atoms on the lattice
    atom_location = np.random.choice(np.arange(N*N), N_atom, replace=False) #pick atom position randomly from NxN array
    atom_location_index = (np.unravel_index(atom_location, (N,N)) - np.ones((2, N_atom))*((N-1)/2)) * M #convert the atom location number to x,y atom location index
    x_index = atom_location_index[0,:] #atoms x location
    y_index = atom_location_index[1,:] #atoms y location

    #Store actual occupation of the atoms for future comparison with the inferred one
    lims = np.arange(0, (N+1)*M, M) - (N*M)/2
    actual_lattice = np.zeros((N, N))
    for ny,nx in product(range(N), range(N)):
        actual_lattice[ny, nx] = np.sum(np.where((x_index > lims[nx]) & (x_index < lims[nx+1]) & (y_index > lims[-(ny+2)]) & (y_index < lims[-(ny+1)]), 1, 0))

    #For each atom sample photons from a Gaussian centered on the lattice site, combine the x,y positions of the counts
    x_loc = np.array([])
    y_loc = np.array([])
    for i in range(N_atom):
        xx, yy = np.random.multivariate_normal([x_index[i], y_index[i]], [[std**2, 0], [0, std**2]], N_photon).T #at each atom location sample N_photons from a Gaussian
        x_loc = np.concatenate((x_loc, xx)) #combine the sampled x-locations for each atom
        y_loc = np.concatenate((y_loc, yy)) #combine the sampled y-locations for each atom

    #Generate dark counts which is the background noise of the camera. Combine dark photon locations with scattered photon locations.
    CCD_x = np.arange(0, N*M, 0.2) - ((N*M)/2) #x-pixel locations
    CCD_y = np.arange(0, N*M, 0.2) - ((N*M)/2) #y-pixel locations
    dark_count=np.random.poisson(N_backg*lam_backg)
    dark_count_location_x = np.random.choice(CCD_x, dark_count, replace=True) #pick a random x location for the dark counts
    dark_count_location_y = np.random.choice(CCD_y, dark_count, replace=True) #pick a random y location for the dark counts
    x_loc = np.concatenate((x_loc, dark_count_location_x)) #combine the sampled x-locations from atoms and dark counts
    y_loc = np.concatenate((y_loc, dark_count_location_y)) #combine the sampled y-locations from atoms and dark counts

    lims = np.arange(0, N*M+1, 1) - (N*M)/2 #pixel
    counts = np.zeros((N*M, N*M)) #pixel
    for ny, nx in product(range(N*M), range(N*M)):
        counts[ny, nx] = np.sum(np.where((x_loc > lims[nx]) & (x_loc < lims[nx+1]) & (y_loc > lims[-(ny+2)]) & (y_loc < lims[-(ny+1)]), 1, 0))

    if plot:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)
        im = plt.imshow(counts, cmap='jet')
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label="Counts", size=16)
        ax.set_xticks(np.arange(0, N*M, M))
        ax.set_yticks(np.arange(0, N*M, M))
        ax.grid(False, color="black")
        #plt.axis('off')
        plt.show()

    return actual_lattice, counts



def run_sweep(N_photon, std, N_back, N, M, lam_back):


    actual_lattice, data_2D = photon_counts(N=N, M=M, N_atom=12, N_photon=N_photon, std=std, N_backg=N_back, lam_backg=lam_back, plot=False)
    actual_lattice_back, data_2D_back = photon_counts(N=N, M=M, N_atom=0, N_photon=N_photon, std=std, N_backg=N_back, lam_backg=lam_back, plot=False)
    x = np.arange(-M/2, M/2)
    X, Y = np.meshgrid(x, x)

    with pm.Model() as mixture_model:

        #Prior
        P = pm.Uniform('P', lower=0, upper=1)
        q = pm.Bernoulli('q', p=P, shape=(N,N)) #(N,N)

        Aa = pm.Uniform('Aa', lower=0.5*np.max(data_2D), upper=np.max(data_2D))#, shape = (N,N))
        Ab = pm.Uniform('Ab', lower=0, upper=np.max(data_2D_back))
        #Ab = pm.Normal('Ab', mu=data_2D_back.mean(), sd=data_2D_back.std())


        sigma_a = pm.Uniform('sigma_a', lower=0, upper=10)
        #sigma_b = pm.Normal('sigma_b', mu=data_2D_back.std(), sd = data_2D_back.std() / (N*M))
        sigma_b = pm.Uniform('sigma_b', lower=0, upper=10)

        atom_std = pm.Normal('std', mu = std, sd = 0.2)
        atom_back = pm.Uniform('A_back', lower=0, upper=20)

        #Model (gaussian + uniform)
        single_background = Ab * np.ones((M, M)) #(M,M)
        single_atom = Aa * np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * atom_std**2)) + atom_back #(M,M)

        atom = tt.slinalg.kron(q, single_atom) #(NM, NM)
        background = tt.slinalg.kron(1-q, single_background) * ((tt.sum(q) + 0)/(N*N)) #(NM, NM)

        #Log-likelihood
        good_data = pm.Normal.dist(mu=atom, sd=sigma_a).logp(data_2D)
        bad_data = pm.Normal.dist(mu=background, sd=sigma_b).logp(data_2D)
        log_like = good_data + bad_data #- 2 * ( P - (tt.sum(q)/(N*N)) )

        pm.Potential('logp', log_like.sum())
        #pm.Potential('const', pm.math.switch(pm.math.eq(tt.sum(q), 0), -1000, 0))

        #Sample
        traces = pm.sample(tune=500, draws=1000, chains=1)
        df = pm.trace_to_dataframe(traces)

    q_array = np.zeros((N,N))

    for ny, nx in product(range(N), range(N)):
        q = "q__{}_{}".format(nx,ny)
        q_array[nx,ny] = df[q].mean()

    fidelity = 100 * (np.sum(actual_lattice==q_array) / N**2)

    return fidelity
