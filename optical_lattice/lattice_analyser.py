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


    def run_analysis(self, analysis_function):
        ''' Initialize empty object

        Parameters
        ----------
        analysis_model : An analysis function of the form
            analysis_func(x, y, std, xsite, ysite)
        '''

        # Retrieve Parameters
        N =  self.generated_lattice_image.N
        M =  self.generated_lattice_image.M
        std = self.generated_lattice_image.std
        x_loc =  self.generated_lattice_image.x_loc
        y_loc =  self.generated_lattice_image.y_loc

        P_array = np.zeros((N,N))
        lims = np.arange(0, (N+1)*M, M) - (N*M)/2 #edges of lattice sites

        # Store center points
        center_points = np.zeros((N, N, 2))

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

                # Store center points
                center_points[nx, ny] = [(xsite[0]+xsite[1])/2, (ysite[0]+ysite[1])/2]

                P_array[ny,nx] = analysis_function(x_new, y_new, std, xsite, ysite)

        #store output
        self.P_array = P_array
        self.center_points = center_points

    def setup_mixture_model(self, pb_lower, pb_upper, sigma=2.5):
        '''TODO: Write Doctring.'''

        # Retrieve Params
        centers = self.generated_lattice_image.center_points
        N =  self.generated_lattice_image.N
        M =  self.generated_lattice_image.M
        x_loc =  self.generated_lattice_image.x_loc
        y_loc =  self.generated_lattice_image.y_loc

        self.mixture_model = pm.Model()
        with self.mixture_model as linear_model:
            Pb = pm.Uniform('Pb', lower=pb_lower, upper=pb_upper)
            q = pm.Bernoulli('q', p=Pb, shape=(N, N))

            # Reformat positions, this array contains entire image data
            positions = [[x_loc[i], y_loc[i]] for i in range(len(y_loc))]

            # Initialize tensors
            gaussian = pm.Normal.dist(mu=centers[0, 0], sd=sigma, shape=(2,) ).logp(positions) * q[0, 0]
            uniform = pm.Uniform.dist(lower = 0, upper = 0.1, shape=(2,)).logp(positions) * (1- q[0, 0])

            # Iterate over lattice_sites
            for nx in range(1, N):
                for ny in range(1, N):
                    gaussian += pm.Normal.dist(mu=centers[nx, ny], sd=sigma, shape=(2,) ).logp(positions) * q[nx, ny]
                    uniform = pm.Uniform.dist(lower = 0, upper = 0.1, shape=(2,)).logp(positions) * (1- q[nx, ny])

            pm.Potential('obs', (gaussian + uniform).sum())

    def sample_mixture_model(self, nsteps=500):
        with self.mixture_model as linear_model:
            traces = pm.sample(tune=nsteps, draws=nsteps, chains=1, init='adapt_diag')

        df = pm.trace_to_dataframe(traces)

        self.mixture_traces = df

    def print_occupation(self):

        #Print the probabilty percentage that lattice site is filled
        np.set_printoptions(precision=1, suppress=True)
        print(self.P_array * 100)
