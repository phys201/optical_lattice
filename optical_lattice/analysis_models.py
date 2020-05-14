"""Define the first generation mixture model."""
import numpy as np

import pymc3 as pm

import theano.tensor as tt

def mixture_model_Boolean_VNM(
        data_2d,
        N,  # noqa: N803
        M,
        std,
        lam_backg,
        nsteps,
        nchains
    ):
    """Define the mixture model and sample from it.
    This version of the model was contributed by
    V N Manoharan

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

    with pm.Model() as mixture_model:

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
        aa = pm.Gamma('Aa', mu=3, sd=0.5)
        ab = pm.Gamma('Ab', mu=0.5, sd=0.1)

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
        single_background = ab * np.ones((M, M)) #(M,M)
        # Replaced background with Ab rather than atom_back.
        single_atom = aa * np.exp(-((X - 0)**2 + (Y - 0)**2) / (2 * atom_std**2)) + Ab * np.ones((M,M)) #atom_back #(M,M)
        
        atom = tt.slinalg.kron(q, single_atom) #(NM, NM)
        background = tt.slinalg.kron(1-q, single_background) #(NM, NM)
        
        #Log-likelihood
        good_data = pm.Normal.dist(mu=atom, sd=sigma_a).logp(data_2d)
        bad_data = pm.Normal.dist(mu=background, sd=sigma_b).logp(data_2d)
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
        
        # Sample
        # sample from the log-likelihood
        traces = pm.sample(tune=nsteps, draws=nsteps, chains=nchains)

    # convert the PymC3 traces into a dataframe
    df = pm.trace_to_dataframe(traces)

    return traces, df

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
    """Define the mixture model and sample from it. This spill over model 
    differs from the above mixture model in that allows fluorescence from
    one site to leak into neighboring sites

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

    # atom center locations are explicitly supplied as the centers of 
    # the lattice sites
    centers = np.linspace(0,(N-1)*M,N)+M/2
    Xcent, Ycent = np.meshgrid(centers,centers)

    with pm.Model() as mixture_model:  # noqa: F841

        # Priors

        # continuous numbers characterizing if lattice sites are filled 
        # or not.
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

        # Instead of tiling a single_atom PSF with kronecker, use 
        # broadcasting and summing along appropriate axis
        # to allow for spill over of one atom to neighboring sites.
        atom = tt.sum(tt.sum(
            q*aa * tt.exp(-((X[:,:,None,None] - Xcent)**2 + 
            (Y[:,:,None,None] - Ycent)**2) / (2 * atom_std**2)),axis=2),axis=2)
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

def mixture_model_mobile_centers(
        data_2d,
        N,  # noqa: N803
        M,
        std,
        lam_backg,
        nsteps,
        nchains
    ):
    """Define the mixture model and sample from it. This mobile centers model 
    extends the above mixture model in that allows the center positions of 
    each atom to vary slightly from the center of the lattice site. This should
    help in cases of lattice inhomogeneity.

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

    # atom center locations are explicitly supplied as the centers of 
    # the lattice sites
    centers = np.linspace(0,(N-1)*M,N)+M/2
    Xcent_mu, Ycent_mu = np.meshgrid(centers,centers)

    with pm.Model() as mixture_model:  # noqa: F841

        # Priors

        # continuous numbers characterizing if lattice sites are filled 
        # or not.
        q = pm.Uniform('q', lower=0, upper=1, shape=(N, N))

        # Allow centers to move but we expect them to be pretty near their lattice centers
        Xcent = pm.Normal('Xcent',mu=Xcent_mu,sigma=Xcent_mu/10,shape=(N,N))
        Ycent = pm.Normal('Ycent',mu=Ycent_mu,sigma=Ycent_mu/10,shape=(N,N))

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

        # Instead of tiling a single_atom PSF with kronecker, use 
        # broadcasting and summing along appropriate axis
        # to allow for spill over of one atom to neighboring sites.
        atom = tt.sum(tt.sum(
            q*aa * tt.exp(-((X[:,:,None,None] - Xcent)**2 + 
            (Y[:,:,None,None] - Ycent)**2) / (2 * atom_std**2)),axis=2),axis=2)
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