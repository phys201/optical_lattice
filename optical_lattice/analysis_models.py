"""Define the first generation mixture model."""
import pymc3 as pm

import numpy as np
import theano.tensor as tt

def mixture_model(data_2D, N, M, std, nsteps, nchains):

    """Define the mixture model and sample from it.


    Parameters
    ----------
    data_2D : ndarray of floats
        2D intensity distribution of the collected light
    N : integer
        number of lattice sites along one axis
    N : integer
        number of pixel per lattice site along one axis
    std : float
        Gaussian width of the point spread function
    nsteps : integer
        The number of samples to draw
    nchains : The number of chains to sample

    Returns
    -------
    traces : pymc3 MultiTrace
        An object that contains the samples.
    df : dataframe
        Samples converted into a dataframe object

    """

    x = np.arange(-M/2, M/2) #x-pixel locations for one lattice site
    X, Y = np.meshgrid(x, x) #X, Y meshgrid of pixel locations

    with pm.Model() as mixture_model:

        #Priors
        P = pm.Uniform('P', lower=0, upper=1) #probability that occupation for the lattice
        q = pm.Bernoulli('q', p=P, shape=(N,N)) #Boolean numbers characterizing if lattice sites is filled or not.

        Aa = pm.Uniform('Aa', lower=0.5*np.max(data_2D), upper=np.max(data_2D)) #Amplitude of the Gaussin signal for the atoms
        Ab = pm.Uniform('Ab', lower=0, upper=10) #Amplitude of the uniform background signal

        sigma_a = pm.Uniform('sigma_a', lower=0, upper=10) #Width of the Gaussian likelihood for the atoms
        sigma_b = pm.Uniform('sigma_b', lower=0, upper=10) #Width of the Gaussian likelihood for the background

        #Model (gaussian + uniform)
        single_atom = Aa * np.exp(-(X**2 + Y**2) / (2 * std**2)) #Gaussian with amplitude Aa modelling the PSF
        atom = tt.slinalg.kron(q, single_atom) #Place a PSF on each lattice site if q=1
        single_background = Ab * np.ones((M, M)) #Uniform distribution with amplitude Ab modelling the background
        background = tt.slinalg.kron(1-q, single_background) #Place a background on each lattice site if q=0

        #Log-likelihood
        good_data = pm.Normal.dist(mu=atom, sd=sigma_a).logp(data_2D) #log-likelihood for the counts to come from atoms
        bad_data = pm.Normal.dist(mu=background, sd=sigma_b).logp(data_2D) #log-likelihood for the counts to come from the background
        log_like = good_data + bad_data

        pm.Potential('logp', log_like.sum())

        #Sample
        traces = pm.sample(tune=nsteps, draws=nsteps, chains=nchains) #sample from the log-likelihood
    df = pm.trace_to_dataframe(traces) #convert the PymC3 traces into a dataframe

    return traces, df

