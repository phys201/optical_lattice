import unittest 
import numpy as np

# TODO remove this import once package is packaged, 
# Move test-file in subdirectory called tests.
from lattice_image import LatticeImage


class TestLatticeImage(unittest.TestCase):
    def test_image_import(self):
        '''Test instanciation of LatticeImage class with sample image'''

        # Some name.
        name = "Testimage"

        # Dimension of atom lattice (in one direction)
        N = 5

        # Pixels per lattice site (in one direction)
        M = 61

        # Path of .image file.
        image_path = "code/static files/test_lattice_image.png"

        # Instanciate LatticeImage class
        lattice_image = LatticeImage(
            name=name,
            M=M,
            N=N,
            image_path=image_path
        )

        # Some easy checks on member variables
        assert lattice_image.name == name
        assert lattice_image.M == M

        # Check if size of image array is correct
        assert np.shape(lattice_image.image) == (N, N, M, M)


if __name__ == '__main__':
    unittest.main() 