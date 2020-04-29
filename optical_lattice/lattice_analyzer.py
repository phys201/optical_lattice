"""Image analysis class."""
import numpy as np


class LatticeImageAnalyzer():
    """Class analyzing generated images with different models."""

    def __init__(self, generated_lattice_image):
        """Initialize empty object.

        Parameters
        ----------
        generated_lattice_image : An instance of
            the GeneratedLatticeImage object.

        Returns
        -------
        P_array = array
            Array of probabilities of each site to be occupied

        """
        # store parameters
        self.generated_lattice_image = generated_lattice_image

    def run_analysis(self, analysis_function):
        """Initialize empty object.

        Parameters
        ----------
        analysis_model : An analysis function of the form
            analysis_func(x, y, std, xsite, ysite)
        """
        # Retrieve Parameters
        N = self.generated_lattice_image.N  # noqa: N806
        M = self.generated_lattice_image.M  # noqa: N806
        std = self.generated_lattice_image.std
        x_loc = self.generated_lattice_image.x_loc
        y_loc = self.generated_lattice_image.y_loc
        lattice_origin = self.generated_lattice_image.lattice_origin

        p_array = np.zeros((N, N))

        # X edges of lattice sites
        xlims = np.arange(-0.5, (N)*M+0.5, M) + lattice_origin[0]

        # Y edges of lattice sites
        ylims = np.arange(-0.5, (N)*M+0.5, M) + lattice_origin[1]

        # Store center points
        center_points = np.zeros((N, N, 2))

        # loop over each site
        for ny in range(N):
            for nx in range(N):
                # if x counts are within that site store them,
                # otherwise equate them to a known number (pi)
                x = np.where(
                    (x_loc > xlims[nx]) & (x_loc <= xlims[nx+1]) &
                    (y_loc > ylims[-(ny+2)]) & (y_loc <= ylims[-(ny+1)]),
                    x_loc,
                    np.pi
                )

                # discard all values equal to the known number (pi)
                x_new = x[x != np.pi]

                # if y counts are within that site store them,
                # otherwise equate them to a known number (pi)
                y = np.where(
                    (x_loc > xlims[nx]) & (x_loc <= xlims[nx+1]) &
                    (y_loc > ylims[-(ny+2)]) & (y_loc <= ylims[-(ny+1)]),
                    y_loc,
                    np.pi
                )

                # discard all values equal to the known number (pi)
                y_new = y[y != np.pi]

                # For each lattice site, select the
                # upper and lower edges along x and y axes
                xsite = np.array([xlims[nx], xlims[nx+1]])
                ysite = np.array([ylims[-(ny+2)], ylims[-(ny+1)]])

                # For each lattice site store the calculated probability value

                # Store center points
                center_points[nx, ny] = [
                    (xsite[0]+xsite[1])/2,
                    (ysite[0]+ysite[1])/2
                ]

                p_array[ny, nx] = analysis_function(
                    x_new,
                    y_new,
                    std,
                    xsite,
                    ysite
                )
                # return (x,y,x_new,y_new,xsite,ysite)

        # store output
        self.P_array = p_array
        self.center_points = center_points

    def print_occupation(self):
        """Print the probabilty percentage that lattice site is filled."""
        np.set_printoptions(precision=1, suppress=True)
        print(self.P_array * 100)
