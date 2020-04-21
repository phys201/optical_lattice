import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import h5py
import gzip
import pickle
from PIL import Image 
import scipy.ndimage as sim

class LatticeImageAnalyzer():
    ''' Class containing all image information of a EMCCD acquired image of a atom lattice'''

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
    
    def _rotate_image(self, image):
        '''Rotate an image ccw by an angle'''
    
        rotated =  sim.rotate(image, angle, reshape=True, mode='wrap')
        
        self.rotated = rotated
        
    def _wiener_deconvolve(self, image, balance = 5e8):
        '''Perform Wiener-Hunt deconvolution given an impulse response'''
        
        deconvolved = restoration.wiener(image, psf, balance)
        deconvolved = deconvolved - np.min(deconvolved)
    
        self.deconvolved = deconvolved
        
    def _shift_image(self, deconvolved_image):
        '''Move an image up and left'''
    
        shifted_image = deconvolved_image[shift_up:, shift_left:] #shift image, first index is up and down, second index is left and right
    
        self.shifted_image = shifted_image
    
    def _find_threshold(self, shifted_image, plot, bins=40):
        '''Histogram the photon counts on lattice sites, extract a binarization threshold'''
    
        N = np.int(shifted_image.shape[1] / M); #number of lattice sites along one axis
        lims = np.arange(0, (N+1)*M, M)
        site_counts = np.zeros((N,N))

        for ny,nx in product(np.arange(0, N), np.arange(0, N)):
                site_counts[ny, nx] = np.sum(shifted_image[lims[ny]:lims[ny+1], lims[nx]:lims[nx+1]])  

        hist_data = site_counts.reshape(site_counts.shape[0]*site_counts.shape[1]);

        y_val, x_val = np.histogram(hist_data, bins=bins);
        max_x = x_val[np.argmax(y_val)]

        array_between_peaks = np.where((x_val[1:] > max_x) & (x_val[1:] < (max_x + 2)), y_val, np.inf)
        threshold = x_val[np.argmin(array_between_peaks)]

        if plot:
            plt.hist(hist_data, bins=bins);
            plt.xlim(1, 5)
            plt.axvline(x = threshold, color='red')

        self.site_counts =  site_counts
        self.threshold = threshold
        
    def _binarize_image(self, site_counts, threshold):
        '''Binarize an image with a threshold, a buffer can be added optionally'''
        
        binarized = np.where(site_counts > threshold + threshold_buffer, 1, 0);
        
        self.binarized = binarized
        
    def _plot_lattice(self):
        '''Plot the lattice pictures'''
    
        fig = plt.figure(figsize=(30, 10))
        plt.tight_layout

        ax = fig.add_subplot(1,3,1)
        plt.imshow(self.raw_img_array, cmap="Blues", interpolation="nearest", vmax=2000);
        ax.set_xticks(np.arange(0, self.raw_img_array.shape[1], M))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0, self.raw_img_array.shape[0], M)) 
        ax.set_yticklabels([])
        ax.grid(True, color="black")

        ax = fig.add_subplot(1,3,2)
        plt.imshow(self.deconvolved, cmap="Blues", interpolation="nearest", vmax=0.06)
        ax.set_xticks(np.arange(0, self.deconvolved.shape[1], M))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0, self.deconvolved.shape[0], M)) 
        ax.set_yticklabels([])
        ax.grid(True, color="black")

        ax = fig.add_subplot(1,3,3)
        plt.imshow(self.binarized, cmap="Blues", interpolation="nearest");
        ax.set_xticks(np.arange(0.5, self.binarized.shape[1], 1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(0.5, self.binarized.shape[0], 1))
        ax.set_yticklabels([])
        ax.grid(True, color="black")
        
        plt.show()
    
    def analyze_raw_data(self, plot, plot_hist):
    
        rotated = rotate_image(raw_img_array)
        rotated_roi = rotated[roi[0]:roi[1], roi[2]:roi[3]]
        deconvolved = wiener_deconvolve(rotated_roi)
        shifted = shift_image(deconvolved)
        site_counts, threshold = find_threshold(shifted, plot=plot_hist)
        binarized = binarize_image(site_counts, threshold)

        if plot:
            plot_lattice(shift_image(rotated_roi), shifted, binarized)

        self.binarized =  binarized

        




