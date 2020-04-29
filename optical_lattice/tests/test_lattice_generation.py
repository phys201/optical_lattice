from unittest import TestCase
import numpy as np

from optical_lattice import GeneratedLatticeImage

class TestLatticeGeneration(TestCase):
    def test_lattice_generation(self):
        N = 3
        M = 10
        N_atom = 5
        N_photon = 100
        std = 1
        N_backg = 100
        lam_backg = 1

        lattice_image = GeneratedLatticeImage(
            N=N,
            M=M,
            N_atom=N_atom,
            N_photon=N_photon,
            std=std,
            N_backg=N_backg,
            lam_backg=lam_backg,
            CCD_resolution=1024
        )

        assert lattice_image.M == M
        assert lattice_image.N == N

        # Check shape of atom positions
        assert  np.shape(lattice_image.actual_lattice) == (N, N)

