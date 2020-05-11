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
