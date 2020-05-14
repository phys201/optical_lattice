"""Define the first generation mixture model."""
import numpy as np

import pymc3 as pm

import theano.tensor as tt


def mixture_model(
        data_2d,
        N,  # noqa: N803
        M,
        std,
        lam_backg,
        nsteps,
        nchains
    ):
    """Define the mixture model and sample from it.

    Parameters
    ----------
    data_2d : ndarray of floats
        2D intensity distribution of the collected light
    N : integer
        number of lattice sites along one axis
    M : integer
        number of pixels per lattice site along one axis
    std : float
        Gaussian width of the point spread function
    lam_backg: integer
        Expected value of the Poissonian background
    nsteps : integer
        number of steps taken by each walker in the pymc3 sampling
    nchains : integer
        number of walkers in the pymc3 sampling

    Returns
    -------
    traces : pymc3 MultiTrace
        An object that contains the samples.
    df : dataframe
        Samples converted into a dataframe object

    """
    # x-pixel locations for one lattice site
    x = np.arange(-M/2, M/2)
    # X, Y meshgrid of pixel locations
    X, Y = np.meshgrid(x, x)  # noqa: N806

    # in future gen instead of passing N, use
    # opticalLatticeShape = tuple((np.array(pixel_grid.shape)/M).astype(int))

    with pm.Model() as mixture_model:  # noqa: F841

        # Priors

        # probability that occupation for the lattice
        P = pm.Uniform('P', lower=0, upper=1)  # noqa: N806

        # Boolean numbers characterizing if lattice sites is filled or not.
        q = pm.Bernoulli('q', p=P, shape=(N, N))

        # Amplitude of the Gaussian signal for the atoms
        aa = pm.Uniform('Aa', lower=0.5*np.max(data_2d), upper=np.max(data_2d))

        # Amplitude of the uniform background signal
        ab = pm.Uniform('Ab', lower=0, upper=lam_backg / (M * M * N))

        # Width of the point spread function
        atom_std = pm.Normal('std', mu=std, sd=0.2)

        # Background offset for the atoms
        atom_back = pm.Uniform('A_back', lower=0, upper=20)

        # Width of the Gaussian likelihood for the atoms
        sigma_a = pm.Uniform('sigma_a', lower=0, upper=10)

        # Width of the Gaussian likelihood for the background
        sigma_b = pm.Uniform('sigma_b', lower=0, upper=10)

        # Model (gaussian + uniform)

        # Gaussian with amplitude Aa modelling the PSF
        single_atom = aa * np.exp(-(X**2 + Y**2) / (2 * atom_std**2)) + \
            atom_back

        # Place a PSF on each lattice site if q=1
        atom = tt.slinalg.kron(q, single_atom)

        # Constant background with amplitude, Ab, drawn from a
        # Uniform distribution, modelling the background
        single_background = ab * np.ones((M, M))

        # Place a background on each lattice site if q=0
        # Penalize the case where all q's are 0
        background = tt.slinalg.kron(1-q, single_background) * \
            (tt.sum(q)/(N*N))

        # Log-likelihood
        # log-likelihood for the counts to come from atoms
        good_data = pm.Normal.dist(mu=atom, sd=sigma_a).logp(data_2d)

        # log-likelihood for the counts to come from the background
        bad_data = pm.Normal.dist(mu=background, sd=sigma_b).logp(data_2d)
        log_like = good_data + bad_data

        pm.Potential('logp', log_like.sum())

        # Sample
        # sample from the log-likelihood
        traces = pm.sample(tune=nsteps, draws=nsteps, chains=nchains)

    # convert the PymC3 traces into a dataframe
    df = pm.trace_to_dataframe(traces)

    return traces, df


def mixture_model_spill_over(
        data_2d,
        N,  # noqa: N803
        M,
        std,
        lam_backg,
        nsteps,
        nchains
    ):
    """Define the mixture model and sample from it. This spill over model differs from the above mixture model
    in that allows fluorescence from one site to leak into neighboring sites

    Parameters
    ----------
    data_2d : ndarray of floats
        2D intensity distribution of the collected light
    N : integer
        number of lattice sites along one axis
    M : integer
        number of pixels per lattice site along one axis
    std : float
        Gaussian width of the point spread function
    lam_backg: integer
        Expected value of the Poissonian background
    nsteps : integer
        number of steps taken by each walker in the pymc3 sampling
    nchains : integer
        number of walkers in the pymc3 sampling

    Returns
    -------
    traces : pymc3 MultiTrace
        An object that contains the samples.
    df : dataframe
        Samples converted into a dataframe object

    """
    # x-pixel locations for the entire image
    x = np.arange(0,N*M)
    # X, Y meshgrid of pixel locations
    X, Y = np.meshgrid(x, x)  # noqa: N806

    # atom center locations are explicitly supplied as the centers of the lattice sites
    centers = np.linspace(0,(N-1)*M,N)+M/2
    Xcent, Ycent = np.meshgrid(centers,centers)

    with pm.Model() as mixture_model:  # noqa: F841

        # Priors

        # continuous numbers characterizing if lattice sites is filled or not.
        q = pm.Uniform('q', lower=0, upper=1, shape=(N, N))

        # Amplitude of the Gaussian signal for the atoms
        aa = pm.Gamma('Aa', mu=3, sd=0.5)
        # Amplitude of the uniform background signal
        ab = pm.Gamma('Ab', mu=0.5, sd=0.1)

        # Width of the Gaussian likelihood for the atoms
        sigma_a = pm.Gamma('sigma_a', mu=1, sd=0.1)

        # Width of the Gaussian likelihood for the background
        sigma_b = pm.Gamma('sigma_b', mu=1, sd=0.1)
        
        # Width of the point spread function
        atom_std = pm.Gamma('std', mu = std, sd = 0.1)

        # Instead of tiling a single_atom PSF with kronecker, use broadcasting and summing along appropriate axis
        # to allow for spill over of one atom to neighboring sites.
        atom = tt.sum(tt.sum(
            q*aa * tt.exp(-((X[:,:,None,None] - Xcent)**2 + (Y[:,:,None,None] - Ycent)**2) / (2 * atom_std**2)),axis=2),axis=2)
        atom += ab
        
        # background is just flat
        background = ab*np.ones((N*M,N*M))
        #Log-likelihood
        good_data = pm.Normal.dist(mu=atom, sd=sigma_a).logp(data_2d)
        bad_data = pm.Normal.dist(mu=background, sd=sigma_b).logp(data_2d)
        log_like = good_data + bad_data
        
        pm.Potential('logp', log_like.sum())
        
        #Sample
        traces = pm.sample(tune=nsteps, draws=nsteps, chains=nchains)

    # convert the PymC3 traces into a dataframe
    df = pm.trace_to_dataframe(traces)

    return traces, df
