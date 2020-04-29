import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from itertools import product

class LatticeImageAnalyzer():

    ''' Class analyzing generated images with a mixture model.'''

    def __init__(self, generated_lattice_image):
        ''' Initialize empty object

        Parameters
        ----------
        generated_lattice_image : An instance of the GeneratedLatticeImage object.


        Returns
        -------
        q_array = array
            Array of probabilities of each site to be occupied

        '''
        #store parameters
        self.generated_lattice_image = generated_lattice_image


    def sample_mixture_model(self, mixture_model, nsteps = 500, nchains = 2):
        ''' Initialize empty object

        Parameters
        ----------
        mixture_model : An analysis function of the form mixture_model(data_2D, N, M, std, nsteps, nchains)
        '''

        # Retrieve Parameters
        data_2D = self.generated_lattice_image.pixel_grid
        M =  self.generated_lattice_image.M
        N =  self.generated_lattice_image.N
        std = self.generated_lattice_image.std

        traces, df = mixture_model(data_2D=data_2D, N = N, M = M, std = std, nsteps=nsteps, nchains=nchains)
        
        self.traces = traces
        self.df = df

    def print_occupation(self):
        N =  self.generated_lattice_image.N #number of lattice sites along one axis
        q_array = np.zeros((N,N)) #for each lattice site there is a q value

        for ny, nx in product(range(N), range(N)): #loop over lattice sites
            q = "q__{}_{}".format(nx,ny)
            q_array[nx,ny] = self.df[q].mean() #take q value for each lattice from samples
        
        self.q_array = q_array 
        print(q_array)
