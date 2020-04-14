import numpy as np
import pymc3 as pm
import theano.tensor as tt

def mixture_model_v0(x, y, std, xsite, ysite):

    with pm.Model() as mixture_model_v0:

        #Prior
        P = pm.Uniform('P', lower=0, upper=1)

        xc = (xsite[0]+xsite[1])/2 #x center of the site
        yc = (ysite[0]+ysite[1])/2 #y center of the site

        #Photons scattered from the atoms are Gaussian distributed
        atom_x = pm.Normal.dist(mu=xc, sigma=std).logp(x)
        atom_y = pm.Normal.dist(mu=yc, sigma=std).logp(y)
        atom = atom_x + atom_y

        #Photons from the camera background are uniform distributed
        background_x = pm.Uniform.dist(lower = xsite[0], upper = xsite[1]).logp(x)
        background_y = pm.Uniform.dist(lower = ysite[0], upper = ysite[1]).logp(y)
        background = background_x + background_y

        #Log-likelihood
        log_like = tt.log((P * tt.exp(atom) + (1-P) * tt.exp(background)))

        pm.Potential('logp', log_like.sum())
    
    #MAP value
    map_estimate = pm.find_MAP(model=mixture_model_v0)
    P_value = map_estimate["P"]
    
    return P_value