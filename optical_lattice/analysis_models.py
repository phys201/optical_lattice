import numpy as np
import pymc3 as pm
import theano.tensor as tt

class AnalysisModels():

    ''' Class containing different models that are used to analyze lattice images.'''

    def __init__(self, x, y, std, xsite, ysite):
        ''' Initialize empty object

        Parameters
        ----------
        x, y : array
            x and y positions of photon counts
        std: float
            standard deviation of the Gaussian that is sampled from
        xsite: array (shape = (1,2))
            lower and upper limits of the lattice site along x axis
        ysite: array (shape = (1,2))
            lower and upper limits of the lattice site along y axis
            
        Returns
        -------
        P_value = float or array
            Probability that a lattice is filled.

        '''
        self.x = x
        self.y = y
        self.std = std
        self.xsite = xsite
        self.ysite = ysite
        
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