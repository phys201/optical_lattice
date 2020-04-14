import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from  itertools import product


class LatticeImage():
    ''' Class containing all image information of a EMCCD acquired image of a atom lattice'''

    
    def __init__(self, name, N, M, image_path):
        ''' Initialize empyt LatticeImage object

        Parameters
        ----------
        name : A reference for this image.
        N : Dimension of optical lattice.
        M : Number of pixels per one side of lattice side.
        jpeg_path: The path of the image file to be loaded.
        '''

        # Store dimensions as member variables.
        self.N = N
        self.M = M 

        # Store name.
        self.name = name

        # Initialize empty numpy array of corresponding length.
        self.image = np.zeros((N, N, M, M))

        # Load raw image
        self._load_from_jpeg(image_path)

        # Update pre-structured self.image array to contian image data
        self._structure_image()

    def _load_from_jpeg(self, image_path):
        '''Load image data from jpeg file'''

        # Load image as greyscale image.
        raw_image = Image.open(image_path).convert('L')

        # Check if image size matches dimension of LatticeImage.
        target_dimension = self.M * self.N 

        if not raw_image.size[0] == target_dimension and raw_image.size[1] == target_dimension:
            error_msg = f"Image dimensions {raw_image.size} does not fit target dimension of ({target_dimension}, {target_dimension})"
            raise Exception(error_msg)

        # Assign raw image to member variable
        self.raw_image = raw_image 

    def _structure_image(self):
        """Load raw data into pre-strucured arra self.image"""

        # Retrieve dimensions for convenience.
        M = self.M
        N = self.N

        # Load data into (M*N) * (M8N) array
        image_array = np.array(self.raw_image)

        # Iterate over lattice sites and fill them with image data.
        for i, j in product(range(N), range(N)):
                self.image[i, j] = image_array[i*M:(i+1)*M, j*M:(j+1)*M]

    def show_raw_image(self):
        '''Show raw image'''
        plt.imshow(np.asarray(self.raw_image))

    def plot_image(self):
        '''Plots the structure image'''

        # Plot the image in a N*N grid
        fig  = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(self.N, self.N)

        # set the spacing between axes. 
        gs.update(wspace=0.001, hspace=0.001) 

        for i, j in product(range(self.N), range(self.N)):
            ax = fig.add_subplot(gs[i,j])
            plt.imshow(self.image[i,j])

            # hide ticks of main axes
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_aspect('equal')

        plt.show()


