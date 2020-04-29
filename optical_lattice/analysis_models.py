"""Define the first generation mixture model."""
import pymc3 as pm

import theano.tensor as tt


def mixture_model_v0(x, y, std, xsite, ysite):
    """Define the mixture model.

    Parameters
    ----------
    x : ndarray of floats
        x positions of the image sensor clicks
    y :
        y positions of the image sensor clicks
    std : float
        Gaussian width of the point spread function
    xsite : ndarray of ints
        Two element array containing the lower
        and upper x edges of the given lattice site
    ysite : ndarray of ints
        Two element array containing the
        lower and upper y edges of the given lattice site
    """
    with pm.Model() as mixture_model_v0:

        # Prior
        p = pm.Uniform('P', lower=0, upper=1)

        xc = (xsite[0]+xsite[1]) / 2  # x center of the site
        yc = (ysite[0]+ysite[1]) / 2  # y center of the site

        # Photons scattered from the atoms are Gaussian distributed
        atom_x = pm.Normal.dist(mu=xc, sigma=std).logp(x)
        atom_y = pm.Normal.dist(mu=yc, sigma=std).logp(y)
        atom = atom_x + atom_y

        # Photons from the camera background are uniform distributed
        background_x = pm.Uniform.dist(
            lower=xsite[0],
            upper=xsite[1]
        ).logp(x)

        background_y = pm.Uniform.dist(
            lower=ysite[0],
            upper=ysite[1]
        ).logp(y)

        background = background_x + background_y

        # Log-likelihood
        log_like = tt.log((p * tt.exp(atom) + (1-p) * tt.exp(background)))

        pm.Potential('logp', log_like.sum())

    # MAP value
    map_estimate = pm.find_MAP(model=mixture_model_v0)
    p_value = map_estimate["P"]

    return p_value
