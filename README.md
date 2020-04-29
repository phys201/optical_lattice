# optical_lattice
Package for site resolution of atoms contained in an optical lattice.

## System
This package employs pymc3 for Markov Chain Monte Carlo (MCMC) sampling of generative mixture models which represent fluorescence data of atoms trapped in an optical lattice.

## Data Used
The raw data is typically a square array of pixel intensities produced by an image sensor. For example, it could be the 1024x1024 intensity map output from a 1 Mega Pixel CCD. 

## Background
In trapped cold atom experiments, ultra cold gasses of atoms are confined by a periodic potential known as an optical lattice. Fluorescence of the atoms is used to image the occupation of these lattice sites by single atoms. Historically, this fluorescence data was processed by deconvolving the point spread function (PSF), an artifact of the imaging system, from the raw image. Then a histogram of intensity at each lattice site would be created, and then a threshold of intensity would set to define occupation of a lattice site. Finally, using this thresholdm the data would be binarized into occupide and unoccupied lattice sites. 

However, there are limitations to this traditional technique. In particular, when working with Erbium, the fluorescence data must be severly undersampled due to technical constraints. So, the raw image is no longer a simple convolution which can be deconvolved. Instead, the MCMC sampling developed in this package can be used to estimate the lattice site occupation. Moreover, the traditional technique is fraught with hidden priors, for example the binarization threshold, and its uncertainties are not easy to characterize. On the other hand, uncertainties are built into the MCMC sampling approach and can be reported along with the lattice site occupation.