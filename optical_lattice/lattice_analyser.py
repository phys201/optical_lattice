import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt

class LatticeImageAnalyzer():

    ''' Class analyzing generated images with different models.'''

    def __init__(self, generated_lattice_image):
        ''' Initialize empty object

        Parameters
        ----------
        generated_lattice_image : An instance of the GeneratedLatticeImage object.

            
        Returns
        -------
        P_array = array
            Array of probabilities of each site to be occupied

        '''
        #store parameters
        self.generated_lattice_image = generated_lattice_image
        

    def run_analysis(self, analysis_function, std):
        ''' Initialize empty object

        Parameters
        ----------
        analysis_model : An analyis function of the form
            analysis_func(x, y, std, xsite, ysite)
        '''

        # Retrieve Parameters
        N =  self.generated_lattice_image.N
        M =  self.generated_lattice_image.M
        x_loc =  self.generated_lattice_image.x_loc
        y_loc =  self.generated_lattice_image.y_loc
    
        P_array = np.zeros((N,N))
        lims = np.arange(0, (N+1)*M, M) - (N*M)/2 #edges of lattice sites
        
        #loop over each site
        for ny in range(N): 
            for nx in range(N):
                #if x counts are within that site store them, otherwise equate them to a known number (pi)
                x = np.where((x_loc > lims[nx]) & (x_loc <= lims[nx+1]) & (y_loc > lims[-(ny+2)]) & (y_loc <= lims[-(ny+1)]), x_loc, np.pi)  
                x_new = x[x != np.pi] #discard all values equal to the known number (pi)
                
                #if y counts are within that site store them, otherwise equate them to a known number (pi)
                y = np.where((x_loc > lims[nx]) & (x_loc <= lims[nx+1]) & (y_loc > lims[-(ny+2)]) & (y_loc <= lims[-(ny+1)]), y_loc, np.pi)
                y_new = y[y != np.pi] #discard all values equal to the known number (pi)
                
                #For each lattice site, select the upper and lower edges along x and y axes
                xsite = np.array([lims[nx], lims[nx+1]])
                ysite = np.array([lims[-(ny+2)], lims[-(ny+1)]])
                
                #For each lattice site store the calculated probability value
                P_array[ny,nx] = analysis_function(x_new, y_new, std, xsite, ysite)
        
        #store output
        self.P_array = P_array
        
    def print_occupation(self):
        
        #Print the probabilty percentage that lattice site is filled
        np.set_printoptions(precision=1, suppress=True)
        print(self.P_array * 100)
        