"""Lattice generation class definition."""
import matplotlib.pyplot as plt

import numpy as np


class GeneratedLatticeImage():
    """A generated data object.

    Attributes:
        N : integer
            number of lattice sites along one direction (NxN)
        M : integer
            number of camera pixels per lattice site along one direction (MxM)
        x_loc : ndarray of ints
            x positions of all photon counts due to CCD
            dark counts and atom fluorescence
        y_loc : ndarray of ints
            y positions of all photon counts due to CCD
            dark counts and atom fluorescence
        actual_lattice : 2d ndarray of ints
            2d grid representing true filling of optical lattice
        pixel_grid : 2d ndarray of ints
            2d grid of CCD intensity values

    """

    def __init__(
        self,
        N,  # noqa: N803
        M,
        N_atom,
        N_photon,
        CCD_resolution,
        std,
        N_backg,
        lam_backg,
        lattice_origin=(0, 0),
    ):
        """Generate lattice image.

        Generates positions of photon counts and CCD intensity values from
        the randomly placed atoms on a lattice and from Poissonian dark counts.
        Includes full CCD data with dedicated optical lattice region.

        Parameters
        ----------
        N : integer
            number of lattice sites along one direction (NxN)
        M : integer
            number of camera pixels per lattice site along one direction (MxM)
        N_atom : integer
            total number of atoms on the lattice
        N_photon : integer
            number of photons sampled from an atom
        CCD_resolution : integer
            number of pixels along one axis of CCD
        lattice_origi n: tuple of integers
            top left corner of optical lattice region.
            Optical lattice region is (M*N)x(M*N) pixels large.
        std : float
            standard deviation of the Gaussian that is sampled from
        N_backg : integer
            number of samples drawn from the Poisson distribution
            for the background noise
        lam_back : float
            expectation interval of the Poisson dark count event

        Returns
        -------
        xloc, yloc : ndarrays
            x and y positions of all the photon counts
        actual_lattice : 2d ndarray of ints
            N*M x N*M grid representing true filling of optical lattice
        pixel_grid : 2d ndarray of ints
            CCD_resolution x CCD resolution grid of CCD intensity values
        """
        # Store Dimensions and std
        self.N = N
        self.M = M
        self.std = std
        self.lattice_origin = lattice_origin

        # Randomly place atoms on the lattice
        # pick atom position randomly from NxN array
        atom_location = np.random.choice(np.arange(N*N), N_atom, replace=False)

        actual_lattice = np.zeros((N, N))
        atom_location_index = np.unravel_index(atom_location, (N, N))


        #Store actual occupation of the atoms for future comparison with the inferred one
        for x,y in zip(atom_location_index[0],atom_location_index[1]):
            # Uncomment for non-inverted y
            #actual_lattice[y,x] = 1

            # Uncomment for inverted y
            actual_lattice[N-y-1,x] = 1


        atom_location_index = atom_location_index + np.zeros((2, N_atom))*M*N #convert the atom location number to x,y atom location index

        #Use M-1 because positions are zero indexed
        atom_location_index = atom_location_index*(M) + ((M-1)/2)
        x_index = atom_location_index[0,:] #atoms x location
        y_index = N*(M) - atom_location_index[1,:] - 1 #atoms y location

        pixel_grid = np.zeros((CCD_resolution,CCD_resolution))

        # For each atom sample photons from a Gaussian centered on
        # the lattice site, combine the x,y positions of the counts
        x_loc = np.array([])
        y_loc = np.array([])
        for i in range(N_atom):
            # at each atom location sample N_photons from a Gaussian
            xx, yy = np.random.multivariate_normal(
                [x_index[i], y_index[i]],
                [
                    [std, 0],
                    [0, std]
                ],
                N_photon
            ).T
            # Round and cast photon positions to respect pixel postions
            xx = np.rint(xx).astype(int) + lattice_origin[0]
            plt.show()
            yy = np.rint(yy).astype(int) + lattice_origin[1]
            x_loc = np.concatenate((x_loc, xx)) #combine the sampled x-locations for each atom
            y_loc = np.concatenate((y_loc, yy)) #combine the sampled y-locations for each atom

        #Generate dark counts which is the background noise of the camera. Combine dark photon locations with scattered photon locations.
        CCD_x = np.arange(0, CCD_resolution, 1) #x-pixel locations
        CCD_y = np.arange(0, CCD_resolution, 1) #y-pixel locations
        dark_count = np.random.poisson(lam_backg, N_backg) #create dark counts sampling from a Poisson distribution, this gives numbers corresponding to number of dark counts
        dark_count_location_x = np.random.choice(CCD_x, np.sum(dark_count), replace=True) #pick a random x location for the dark counts
        dark_count_location_y = np.random.choice(CCD_y, np.sum(dark_count), replace=True) #pick a random y location for the dark counts

        x_loc = np.concatenate((x_loc, dark_count_location_x)).astype(int) #combine the sampled x-locations from atoms and dark counts
        y_loc = np.concatenate((y_loc, dark_count_location_y)).astype(int) #combine the sampled y-locations from atoms and dark counts

        #convert counts to intensity values for each pixel

        for x,y in zip(x_loc,y_loc):
            if(x<CCD_resolution and y<CCD_resolution and x>=0 and y>=0):
                # pixel_grid[CCD_resolution-y-1,x] += 1
                pixel_grid[y, x] += 1

        #Shift and scale counts to mimic data from CCD
        #pixel_grid *= 5000/pixel_grid.max()
        #pixel_grid += 600

        # Store output results
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.actual_lattice = actual_lattice
        self.pixel_grid = pixel_grid

    # useful for checking that each count sits in the middle of a pixel
    # isolates a particular part of the generated photon counts to more
    # easily see each pixel
    def grid_plot(self, num_sites=1, invert_y=False):
        """Plot the image (collected photons) on the camera."""
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        xlims = (self.lattice_origin[0], self.lattice_origin[0]+self.N*self.M)
        ylims = (self.lattice_origin[1], self.lattice_origin[1]+self.N*self.M)
        coords = np.zeros((2, len(self.x_loc)), dtype=int)
        coords[0, :] = self.x_loc
        coords[1, :] = self.y_loc
        coords = coords.T
        lattice_coords = coords[(coords[:,0]>xlims[0])*(coords[:,0]<xlims[1])*(coords[:,1]>ylims[0])*(coords[:,1]<ylims[1]),:]
        reduced_lattice_coords = lattice_coords[(lattice_coords[:,0] < (num_sites*self.M + xlims[0]))*(lattice_coords[:,1] < (num_sites*self.M + ylims[0]))]
        #im = plt.plot(reduced_lattice_coords.T[0], reduced_lattice_coords.T[1], 'ko', markersize=1,alpha=0.25) #plot counts
        im = plt.imshow(self.pixel_grid)
        fig.colorbar(im, ax=ax)
        # grid lines outline pixel locations
        #ax.set_xticks(np.arange(-0.5-1*self.M, (num_sites+1)*self.M+0.5, 1) + xlims[0]) #vertical lines as visual aid
        #ax.set_yticks(np.arange(-0.5-1*self.M, (num_sites+1)*self.M+0.5, 1) + ylims[0]) #horizontal lines as visual aid
        ax.set_xticks(np.arange(0, self.N*self.M, self.M))
        ax.set_yticks(np.arange(0, self.N*self.M, self.M))
        ax.grid(True, color="black")

        if(invert_y):
            ax.invert_yaxis()


    def plot(self,invert_y = False,alpha=0.25,markersize=0.1):
        '''Plot the image (collected photons) on the camera.'''
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1,1,1)
        im = plt.plot(self.x_loc, self.y_loc, 'k.', markersize=markersize,alpha=alpha) #plot counts
        # grid lines outline optical lattice sites
        ax.set_xticks(np.arange(-0.5-1*self.M, (self.N+1)*self.M+0.5, self.M) + self.lattice_origin[0]) #vertical lines as visual aid
        ax.set_yticks(np.arange(-0.5-1*self.M, (self.N+1)*self.M+0.5, self.M) + self.lattice_origin[1]) #horizontal lines as visual aid
        ax.grid(True, color="red")
        if(invert_y):
            ax.invert_yaxis()
