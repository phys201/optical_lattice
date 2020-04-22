from unittest import TestCase
import numpy as np

from optical_lattice import ConvolutionLatticeImageAnalyzer


class TestLatticeImageAnalyzer(TestCase):
    def test_instanciation(self):

        raw_image_path = 'optical_lattice/data/Scan-20200201-0006.hdf'
        psf_path = 'optical_lattice/data/20191216-0030.pkl'
        shot_number = 4
        M = 10 #Number of pixels per lattice site
        angle = 47.5 #Angle by which the raw image is rotated
        roi = [650, 1050, 720, 1120] #[x1,x2,y1,y2] Region of interest of the raw image
        shift_up = 5 #Number of pixels by which the deconvolved image is shifted up such that atom locations match the lattice sites
        shift_left = 0 #Number of pixels by which the deconvolved image is shifted left such that atom locations match the lattice sites
        threshold_buffer = 0.5 #Threshold buffer to be added to the calculated threshold for binarizing the deconvolved image

        single_site_image = ConvolutionLatticeImageAnalyzer(
            raw_image_path = raw_image_path,
            shot_number = shot_number,
            psf_path = psf_path,
            M=M,
            angle=angle,
            roi=roi,
            shift_up=shift_up,
            shift_left=shift_left,
            threshold_buffer=threshold_buffer
        )

