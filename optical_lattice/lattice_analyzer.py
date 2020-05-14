"""Image analysis class."""
from itertools import product

import matplotlib.pyplot as plt

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
        lam_backg = self.generated_lattice_image.lam_backg

        traces, df = mixture_model(
            data_2d=data_2d,
            N=N,
            M=M,
            std=std,
            lam_backg=lam_backg,
            nsteps=nsteps,
            nchains=nchains
        )

        self.traces = traces
        self.df = df

    def print_occupation(self):
        """Transform qi arrray into occuptation numbers."""
        # number of lattice sites along one axis
        N = self.generated_lattice_image.N  # noqa: N806
        actual_lattice = self.generated_lattice_image.actual_lattice
        q_array = np.zeros((N, N))  # for each lattice site there is a q value

        # loop over lattice sites
        for ny, nx in product(range(N), range(N)):
            q = "q__{}_{}".format(nx, ny)
            # take q value for each lattice from samples
            q_array[nx, ny] = self.df[q].mean()

        np.set_printoptions(precision=3)
        print(q_array)
        self.q_array = q_array
        
    def plot_histogram(self):
        """Histogram of mean q values"""
        plt.figure(figsize=(6,4))
        plt.hist(self.q_array.flatten(), bins=20, color="blue");
        plt.xlabel("Mean q", fontsize=16);
        plt.ylabel("Counts", fontsize=16)

    def plot_occupation(self, threshold):
        """Plot the inferred occupation probability of atoms."""
        # Retrieve Parameters
        data_2d = self.generated_lattice_image.pixel_grid
        actual_lattice = self.generated_lattice_image.actual_lattice
        N = self.generated_lattice_image.N  # noqa: N806
        # Binarize the image with the threshold
        binarized = np.where(self.q_array > threshold, 1, 0)
        # Calculate the fidelity
        fidelity = 100 * (np.sum(binarized == actual_lattice) / N**2)

        fig = plt.figure(figsize=(24, 6))
        plt.tight_layout()
        
        ax = fig.add_subplot(1, 4, 1)
        im = plt.imshow(actual_lattice, cmap='Greys');
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        ax.set_xticks(np.arange(0.49, N, 1))
        ax.xaxis.set_ticklabels([])
        ax.set_yticks(np.arange(0.49, N, 1)) 
        ax.yaxis.set_ticklabels([])
        ax.grid(True, color="black")
        plt.title('Actual Lattice')

        ax = fig.add_subplot(1, 4, 2)
        im = plt.imshow(data_2d, cmap="jet", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label="Counts", size=16)
        plt.axis('off')
        plt.title('Generated Data')

        ax = fig.add_subplot(1, 4, 3)
        im = plt.imshow(self.q_array, cmap='seismic');
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label=r"$\langle q \rangle$", size=16)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        ax.set_xticks(np.arange(0.49, N, 1))
        ax.xaxis.set_ticklabels([])
        ax.set_yticks(np.arange(0.49, N, 1)) 
        ax.yaxis.set_ticklabels([])
        ax.grid(True, color="black")
        plt.title('Inferred Occupation Fractions')
        
        ax = fig.add_subplot(1, 4, 4)
        im = plt.imshow(binarized, cmap='Greys')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        ax.set_xticks(np.arange(0.49, N, 1))
        ax.xaxis.set_ticklabels([])
        ax.set_yticks(np.arange(0.49, N, 1)) 
        ax.yaxis.set_ticklabels([])
        ax.grid(True, color="black")
        plt.title(r'Fidelity = %.2f Percent ' %fidelity)
        plt.tight_layout
        plt.show()
