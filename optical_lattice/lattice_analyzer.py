"""Image analysis class."""
from itertools import product

import numpy as np


class LatticeImageAnalyzer():
    """Class analyzing generated images with  a mixture model."""

    def __init__(self, generated_lattice_image):
        """Initialize empty object.

        Parameters
        ----------
        generated_lattice_image : An instance of
            the GeneratedLatticeImage object.

        Returns
        -------
        q_array = array
            Array of probabilities of each site to be occupied

        """
        # store parameters
        self.generated_lattice_image = generated_lattice_image

    def sample_mixture_model(self, mixture_model, nsteps=500, nchains=2):
        """Run an Markov Chain Monte Carlo sampling of the given mixture_model.

        Parameters
        ----------
        mixture_model : An analysis function of the form
            mixture_model(data_2D, N, M, std, nsteps, nchains)
        nsteps : integer
            number of steps taken by each walker in the MCMC sampling
        nchains : integer
            number of walkers in the MCMC sampling
        """
        # Retrieve Parameters
        data_2d = self.generated_lattice_image.pixel_grid
        M = self.generated_lattice_image.M  # noqa: N806
        N = self.generated_lattice_image.N  # noqa: N806
        std = self.generated_lattice_image.std

        traces, df = mixture_model(
            data_2d=data_2d,
            N=N,
            M=M,
            std=std,
            nsteps=nsteps,
            nchains=nchains
        )

        self.traces = traces
        self.df = df

    def print_occupation(self):
        """Transform qi arrray into occuptation numbers."""
        # number of lattice sites along one axis
        N = self.generated_lattice_image.N  # noqa: N806
        q_array = np.zeros((N, N))  # for each lattice site there is a q value

        # loop over lattice sites
        for ny, nx in product(range(N), range(N)):
            q = "q__{}_{}".format(nx, ny)
            # take q value for each lattice from samples
            q_array[nx, ny] = self.df[q].mean()

        self.q_array = q_array
        print(q_array)
