import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from itertools import product
import seaborn as sns


def photon_counts(N, M, N_atom, N_photon, std, lam_backg, plot=True):
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
    #atom_location = 5
    atom_location_index = (np.unravel_index(atom_location, (N,N)) - np.ones((2, N_atom))*((N-1)/2)) * M #convert the atom location number to x,y atom location index
    x_index = atom_location_index[0,:] #atoms x location
    y_index = atom_location_index[1,:] #atoms y location

    #Store actual occupation of the atoms for future comparison with the inferred one
    lims = np.arange(0, (N+1)*M, M) - (N*M)/2
    actual_lattice = np.zeros((N, N));
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
    CCD_x = np.arange(0, N*M, 0.5) - ((N*M)/2) #x-pixel locations
    CCD_y = np.arange(0, N*M, 0.5) - ((N*M)/2) #y-pixel locations
    dark_count=np.random.poisson(lam_backg)
    dark_count_location_x = np.random.choice(CCD_x, dark_count, replace=True) #pick a random x location for the dark counts
    dark_count_location_y = np.random.choice(CCD_y, dark_count, replace=True) #pick a random y location for the dark counts
    x_loc = np.concatenate((x_loc, dark_count_location_x)) #combine the sampled x-locations from atoms and dark counts
    y_loc = np.concatenate((y_loc, dark_count_location_y)) #combine the sampled y-locations from atoms and dark counts

    lims = np.arange(0, N*M+1, 1) - (N*M)/2 #pixel
    counts = np.zeros((N*M, N*M)); #pixel
    for ny, nx in product(range(N*M), range(N*M)):
        counts[ny, nx] = np.sum(np.where((x_loc > lims[nx]) & (x_loc < lims[nx+1]) & (y_loc > lims[-(ny+2)]) & (y_loc < lims[-(ny+1)]), 1, 0))

    if plot:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1,1,1)
        im = plt.imshow(counts, cmap='jet');
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



def run_sweep(N_photon, std, N, M, lam_back, real_mixture=False):


    actual_lattice, data_2D = photon_counts(N=N, M=M, N_atom=12, N_photon=int(N_photon), std=std, lam_backg=lam_back, plot=True)
    x = np.arange(-M/2, M/2)
    X, Y = np.meshgrid(x, x)

    if not real_mixture:

        # Run our model.

        with pm.Model() as mixture_model:

            #Prior
            q = pm.Uniform('q', lower=0, upper=1, shape=(N,N))

            Aa = pm.Uniform('Aa', lower=0.5*np.max(data_2D), upper=np.max(data_2D))#, shape = (N,N))
            Ab = pm.Uniform('Ab', lower=0, upper=lam_back / (M*M*N))

            sigma_a = pm.Uniform('sigma_a', lower=0, upper=5)
            sigma_b = pm.Uniform('sigma_b', lower=0, upper=5)

            atom_std = pm.Normal('std', mu = std, sd = 0.2)

            #Model (gaussian + uniform)
            single_background = Ab * np.ones((M, M)) #(M,M)
            single_atom = Aa * np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * atom_std**2)) + Ab * np.ones((M, M))

            atom = tt.slinalg.kron(q, single_atom) #(NM, NM)
            background = tt.slinalg.kron(1-q, single_background) * (tt.sum(q)/(N*N)) #(NM, NM)

            #Log-likelihood
            good_data = pm.Normal.dist(mu=atom, sd=sigma_a).logp(data_2D)
            bad_data = pm.Normal.dist(mu=background, sd=sigma_b).logp(data_2D)
            log_like = good_data + bad_data #- 2 * ( P - (tt.sum(q)/(N*N)) )

            pm.Potential('logp', log_like.sum())

            #Sample
            traces = pm.sample(tune=500, draws=750, chains=1)
            df = pm.trace_to_dataframe(traces)


        q_array = np.zeros((N,N))

        for ny, nx in product(range(N), range(N)):
            q = "q__{}_{}".format(nx,ny)
            q_array[nx,ny] = df[q].mean()

        # Normalize qs to be between 0 and 1
        q_array -= np.min(np.abs(q_array),axis=0)
        q_array /= np.max(np.abs(q_array),axis=0)


        binarized = np.where(q_array > 0.5, 1, 0) #threshold from the histogram
        fidelity = 100 * (np.sum(binarized == actual_lattice) / N**2)

    elif real_mixture:

        # Run Vinny's model.
        with pm.Model() as mixture_model_VINNY:

            #Prior
            # Use an informative prior for P based on what you would know in a real experiment.
            # A Uniform(0,1) prior causes severe problems and probably doesn't represent your
            # true state of knowledge prior to the experiment.  I use a Gamma distribution (rather
            # than a Normal) so that P stays positive and the sampler doesn't diverge.  You can
            # adjust the width to match what you would know in a typical experiment.
            P = pm.Gamma('P', mu=0.5, sd=0.05)
            q = pm.Bernoulli('q', p=P, shape=(N,N), testval=np.ones((N,N))) #(N,N)

            # Here again you need more informative priors.  Previously these were Uniform, with
            # limits determined by the data.  But priors should not be based on the data.  They should
            # be based on what you know prior to to experiment.  I use a Gamma distribution for both
            # because that constrains the values to be positive.  Adjust mu and sd to match what you
            # would know before a typical experiment.
            Aa = pm.Gamma('Aa', mu=3, sd=0.5)
            Ab = pm.Gamma('Ab', mu=0.5, sd=0.1)

            # Again, replaced Uniform priors by Gamma priors.  Adjust mu and sd to match what you
            # would know before a typical experiment
            sigma_a = pm.Gamma('sigma_a', mu=1, sd=0.1)
            sigma_b = pm.Gamma('sigma_b', mu=1, sd=0.1)

            # Replaced Normal by Gamma distribution to keep atom_std positive
            #atom_std = pm.Normal('std', mu = std, sd = 0.2)
            atom_std = pm.Gamma('std', mu = std, sd = 0.1)
            # Removed atom_back as a parameter and assumed background in presence of atom is the
            # same as that without the atom.  If you want to keep this, don't use a Uniform prior.
            # atom_back = pm.Uniform('A_back', lower=0, upper=20)

            #Model (gaussian + uniform)
            single_background = Ab * np.ones((M, M)) #(M,M)
            # Replaced background with Ab rather than atom_back.
            single_atom = Aa * np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * atom_std**2)) + Ab * np.ones((M,M)) #atom_back #(M,M)

            atom = tt.slinalg.kron(q, single_atom) #(NM, NM)
            background = tt.slinalg.kron(1-q, single_background) #(NM, NM)

            #Log-likelihood
            good_data = pm.Normal.dist(mu=atom, sd=sigma_a).logp(data_2D)
            bad_data = pm.Normal.dist(mu=background, sd=sigma_b).logp(data_2D)
            log_like = good_data + bad_data

            # Here I added a binomial log-likelihood term.  I used the normal approximation to the
            # binomial (please check my math).  This term accounts for deviations from the expected
            # occupancy fraction.  If the mean of the q_i are signficantly different from P, the
            # configuration is penalized.  This is why you shouldn't put a uniform prior on P.
            pm.Potential('logp', log_like.sum() + pm.Normal.dist(mu=P, tau=N*N/(P*(1-P))).logp(q.mean()))

            #Sample
            # We'll explicitly set the two sampling steps (rather than let pymc3 do it for us), so that
            # we can tune each step.  We use binary Gibbs Metropolis for the q and NUTS for everything
            # else.  Note that if you add a variable to the model, you should explicitly add it to the
            # sampling step below.
            steps = [pm.BinaryGibbsMetropolis([q], transit_p=0.8),
                    pm.NUTS([atom_std, sigma_b, sigma_a, Ab, Aa, P], target_accept=0.8)]

            # Also we'll set the random seed to 0 to make it easier to diagnose problems with the sampler
            traces = pm.sample(tune=500, draws=500, chains=1, step=steps)
            df = pm.trace_to_dataframe(traces)

            q_array = np.zeros((N,N))

            for ny, nx in product(range(N), range(N)):
                q = "q__{}_{}".format(nx,ny)
                q_array[nx,ny] = df[q].mean()

            # note that this definition of fidelity doesn't work when
            # you use the mean of the q_i's, which can be real-valued
            fidelity = 100 * (np.sum(actual_lattice==q_array) / N**2)

    return fidelity


if __name__ == "__main__":

    # Average Photons per atom

    # How many sweep values per axis to evaulate (resulting plot will be num_sweeps * num_sweeps)
    num_sweeps = 3

    # How often to average on each point
    num_average = 1

    # Wheter to use Vinny's model or ours
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

    for i in range(num_sweeps):
        for j in range(num_sweeps):
            n_phot = n_photons[i]
            std = stds[j]

            fidelities = np.zeros(num_average)
            for k in range(num_average):
                fidelities[k] = run_sweep(
                    N_photon=n_phot,
                    std=std,
                    N=N,
                    M=M,
                    lam_back=lam_back,
                    real_mixture=vinnies_model
                )

                print(f"{(i*(num_sweeps *  num_average) + j*num_average + k + 1)/(num_sweeps**2*num_average) * 100:.2f} % done")

            # store average fidelities std std
            fidelities_averages[i, j] = np.average(fidelities)
            fidelities_std_average[i, j] = np.std(fidelities)

            # Save results after each iteration
            np.savetxt('2d_fidelity_average_sweep.txt', fidelities_averages)
            np.savetxt('2d_fidelity_std_sweep.txt', fidelities_std_average)

    df = pd.DataFrame(fidelities_averages, columns=n_photons/num_backgound )
    df = df.set_index(stds/10)

    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(df, vmin=0, vmax=1, cbar_kws={'label': 'Fidelity in %'})
    #ax.set_xlabel('average # photons of atom / average # photons background')
    #ax.set_ylabel('PSF width in lattice sites')
    plt.savefig('av_fidelities_sweep.pdf')
