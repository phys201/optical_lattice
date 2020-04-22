import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import h5py
import gzip
import pickle
from PIL import Image
import scipy.ndimage as sim
from skimage import restoration, measure


class ConvolutionLatticeImageAnalyzer():
    ''' Class loading raw lattice images and PSF, and analyzing them with Wiener-Hunt deconvolution '''

    def __init__(self, raw_image_path, shot_number, psf_path, M, angle, roi, shift_up, shift_left, threshold_buffer):
        ''' Initialize empty LatticeImageAnalyzer object

        Parameters
        ----------
        raw_image_path: The path of the lattice image file to be loaded.
        shot_number: The shot to be analyzed in the lattice image file.
        psf_path: The path of the point spread function file to be loaded.
        '''

        # Store dimensions as member variables.
        self.M = M
        self.angle = angle
        self.roi = roi
        self.shift_up = shift_up
        self.shift_left = shift_left
        self.threshold_buffer = threshold_buffer

        #Load lattice
        self._import_lattice(raw_image_path, shot_number)
        #Load psf
        self._import_PSF(psf_path)

    def _import_lattice(self, raw_image_path, shot_number):
        '''Load lattice image from an hdf file, convert into a numpy array and select a shot for analysis'''

        file = h5py.File(raw_image_path, 'r')
        img = file['Shot-000' + str(shot_number)]['Cameras']['IStar']['Image-0000'] # where Shot-XXXX = 0000, ...., 0005
        raw_img_array = img[()] # convert the data set into a numpy array

        self.raw_img_array = raw_img_array


    def _import_PSF(self, psf_path):
        '''Load psf image from a pkl file, convert it into a numpy array'''

        with open(psf_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()

        avgPSF = p['avgPSF']
        psf = measure.block_reduce(avgPSF, block_size=(p['PSFmag'], p['PSFmag']), func=np.mean) #demagnify the PSF

        self.psf = psf

    def _rotate_image(self, image, angle):
        '''Rotate an image ccw by an angle'''

        rotated =  sim.rotate(image, angle, reshape=True, mode='wrap')

        return rotated

    def _wiener_deconvolve(self, image, psf, balance = 5e8):
        '''Perform Wiener-Hunt deconvolution given an impulse response'''

        deconvolved = restoration.wiener(image, psf, balance)
        deconvolved = deconvolved - np.min(deconvolved)

        return deconvolved

    def _shift_image(self, deconvolved_image, shift_up, shift_left):
        '''Move an image up and left'''

        shifted_image = deconvolved_image[shift_up:, shift_left:] #shift image, first index is up and down, second index is left and right

        return shifted_image

    def _find_threshold(self, shifted_image, M, plot, bins=40):
        '''Histogram the photon counts on lattice sites, extract a binarization threshold'''

        N = np.int(shifted_image.shape[1] / M); #number of lattice sites along one axis
        lims = np.arange(0, (N+1)*M, M) #position of lattice sites
        site_counts = np.zeros((N,N)) #photon counts of lattice sites

        for ny,nx in product(np.arange(0, N), np.arange(0, N)):
                site_counts[ny, nx] = np.sum(shifted_image[lims[ny]:lims[ny+1], lims[nx]:lims[nx+1]]) #goes through lattice sites, adds photon counts at each pixel in the lattice site

        hist_data = site_counts.reshape(site_counts.shape[0]*site_counts.shape[1]); #histogram of photon counts for each lattice site

        y_val, x_val = np.histogram(hist_data, bins=bins);
        max_x = x_val[np.argmax(y_val)] #x value of the maximum count

        array_between_peaks = np.where((x_val[1:] > max_x) & (x_val[1:] < (max_x + 2)), y_val, np.inf) #histogram has two peaks: for an empty site and for a filled site, threshold is in between them
        threshold = x_val[np.argmin(array_between_peaks)] #find the x value of the minimum in between two peaks

        if plot:
            plt.hist(hist_data, bins=bins);
            plt.xlim(1, 5)
            plt.axvline(x = threshold, color='red')
            plt.xlabel('Photon Counts')
            plt.ylabel('Counts')

        return site_counts, threshold

    def _binarize_image(self, site_counts, threshold, threshold_buffer):
        '''Binarize an image with a threshold, a buffer can be added optionally'''

        binarized = np.where(site_counts > threshold + threshold_buffer, 1, 0); #a buffer is added to get more conservative filling results

        return binarized

    def _plot_lattice(self, raw_image, deconvolved_image, binarized_image, M):

        fig = plt.figure(figsize=(30, 10))
        plt.tight_layout

        ax = fig.add_subplot(1,3,1)
        plt.imshow(raw_image, cmap="Blues", interpolation="nearest", vmax=2000);
        ax.set_xticks(np.arange(0, raw_image.shape[1], M))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0, raw_image.shape[0], M))
        ax.set_yticklabels([])
        ax.grid(True, color="black")
        plt.title('Raw Data')

        ax = fig.add_subplot(1,3,2)
        plt.imshow(deconvolved_image, cmap="Blues", interpolation="nearest", vmax=0.06)
        ax.set_xticks(np.arange(0, deconvolved_image.shape[1], M))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0, deconvolved_image.shape[0], M))
        ax.set_yticklabels([])
        ax.grid(True, color="black")
        plt.title('Deconvolved Data')


        ax = fig.add_subplot(1,3,3)
        plt.imshow(binarized_image, cmap="Blues", interpolation="nearest");
        ax.set_xticks(np.arange(0.5, binarized_image.shape[1], 1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0.5, binarized_image.shape[0], 1))
        ax.set_yticklabels([])
        ax.grid(True, color="black")
        plt.title('Site Occupations')

        plt.show()

    def analyze_raw_data(self, plot, plot_hist):

        rotated = self._rotate_image(self.raw_img_array, self.angle)
        rotated_roi = rotated[self.roi[0]:self.roi[1], self.roi[2]:self.roi[3]]
        deconvolved = self._wiener_deconvolve(rotated_roi, self.psf)
        shifted = self._shift_image(deconvolved, self.shift_up, self.shift_left)
        site_counts, threshold = self._find_threshold(shifted, self.M, plot=plot_hist)
        binarized = self._binarize_image(site_counts, threshold, self.threshold_buffer)

        if plot:
            self._plot_lattice(self._shift_image(rotated_roi, self.shift_up, self.shift_left), shifted, binarized, self.M)

        return binarized




