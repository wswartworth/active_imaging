import numpy as np
from itertools import product
from sklearn import neighbors

def _sq_distances(shape, point):
	'''
	Returns an array of size shape, giving the squared distance of each point in the
	grid from point.
	'''
    distances = np.array([0])
    for i in range(0,shape.size):
        new_shape = np.ones(shape.shape, dtype=int)
        new_shape[i] = -1
        distances = (((np.arange(0,shape[i],1) - point[i])**2)).reshape(new_shape) + distances
    return distances

def _mesh(shape):
	'''
	Returns the list of coordinates in a grid of a given shape
	'''
    return np.array(list(product(*[range(0,d) for d in shape])))

def _normalize(x):
    return x/np.linalg.norm(x)

class Sampler:

	"""
    A class used to adaptively sample measurement points, and to reconstruct the
    image via those measurements.
 
    Methods
    -------
    get_next_sample()
        request the next point to sample

    lattice_sample_taken(ind, series)
    	update the sampler from a measurement (typically used in conjunction with 
    	get_next_sample())

    sample_count()
    	gets the number of samples taken

    get_final_reconstruction()
    	gets the fully reconstructed image.  Usually called after all samples have
    	been taken

    get_sample_mask()
    	gets the mask showing all sampled points
    """
    
    def __init__(self, dimensions, grad_weight=0.5, reconstruction_freq=2000, n_neigbor_count=4):

    	"""
        Parameters
        ----------
        dimensions: np.array
            The dimensions of the image.  The number of time series components is given last
        grad_weight : float, optional
            The amount of weight given to relative gradients (default is 0.5)
        reconstruction_freq : int, optional
            The number of iterations between reconstructions (default is 2000)
        n_neighbor_count: int, optional
        	The number of neighbors used for k-nearest-neighbor reconstruction (default is 4)
        """

        self.xyv_dimensions = dimensions[0:-1]
        self.time_dimension = dimensions[-1]

        self.grad_weight = grad_weight
        self.reconstruction_freq = reconstruction_freq
        self.n_neigbor_count = n_neigbor_count

        #tensor of measurements, including time-series data
        self.measurement_tensor = np.empty(dimensions)
        self.measurement_tensor.fill(np.nan)
        
        #mask of samples taken, samples indicated with np.inf
        self.sample_mask = np.zeros(self.xyv_dimensions)

        #corresponding lists of measured coordinate indices, and the mean values
        #of time-series measurements
        self.measured_coords = []
        self.measured_vals = []

        self.num_samples = 0
        
        #functions used to calculate the relative isolation of points
        self.epsilon = 1e-6
        self.f = lambda x: np.exp(-x) #input x will be a squared distance
        self.g = lambda x: 1/(x+self.epsilon)

        #a heat map measuring the isolation of each spatial/volatage point
        self.farness_map = np.zeros(self.xyv_dimensions)

        #A map of the radially-symmetric function associated to f. Used to save
        #computation costs when updating farness_map
        self.big_distance_map = self.f(_sq_distances(4*np.array(self.xyv_dimensions) + 1, 
                                               2*np.array(self.xyv_dimensions)))
        
        #the list of all spatial/volatage coordinates
        self.grid = _mesh(self.xyv_dimensions)

        #the current reconstruction, used for the purposes of generating samples
        self.reconstruction = None
                    
    def get_next_sample(self):

    	"""
		Get the spatial/voltage coordinates of the next point that should be sampled.

    	Returns
    	----------
    	tuple
    		the spatial/voltage coordinates of the next point to sample
    	"""

         if self.num_samples == 0:
              mid = tuple(np.round(np.array(self.xyv_dimensions)/2).astype(int))
              return mid
         else:
              return self._choose_sample()

    def lattice_sample_taken(self, ind, series):

    	"""
		Call this function each time a measurement is taken.  Usually used 
		in conjunction with get_next_sample() although this is not required.

    	Parameters
    	----------
    	ind: tuple
    		The spatial/voltage coordinates of the measurement
    	series: np.array
    		The time series measured
    	"""

         self.measurement_tensor[ind] = series

         measurement = np.mean(series)
         self.sample_mask[ind] = np.inf
         self.measured_coords.append(ind)
         self.measured_vals.append(measurement)
         self.num_samples += 1
         self._update_from_sample(ind)
    
    def sample_count(self):
    	"""
		Gets the number of measurements taken (a time series measurement counts as
		a single measurement).

    	Returns
    	----------
    	int
    		The number of measurements taken
    	"""
        return self.num_samples


    def get_final_reconstruction(self):

    	"""
		Reconstruct the entire image

    	Returns
    	----------
    	np.array
    		The (typically 4D) tensor of time series reconstructions
    	"""

        slice_reconstructions = []
        for t in tqdm(range(self.time_dimension)):

            vals = [self.measurement_tensor[tuple(list(ind)+[t])] 
                   for ind in self.measured_coords]

            slice_reconstructions.append(
                   self._image_reconstruct(self.measured_coords, vals))

        return np.stack(slice_reconstructions, axis=3)

	def get_sample_mask(self):

		"""
		Get the mask consisting of the sampled points in spatial/voltage coordinates

    	Returns
    	----------
    	np.array
    		A tensor indicating the coordinates of each measurement.  Measured coordinates
    		are indicated with np.inf, all other coordinates are 0.
    	"""
        return self.sample_mask
    
    def _choose_sample(self):
    	"""
    	Chooses a sample using a convex combination of relative gradient magnitude,
        and relative isolation.
    	"""

   	    #periodically generate a new reconstruction for the purposes of sampling
        if (self.num_samples % self.reconstruction_freq == 0) or self.reconstruction is None:
            self.reconstruction = self._reconstruct()
            
            self.grad_norms = normalize(np.linalg.norm(  #normalized!
                np.gradient(self.reconstruction), axis=0))
            
        farness = normalize(self.g(self.farness_map)) #normalized!
            
        sample_heur = self.grad_weight*self.grad_norms + (1-self.grad_weight)*farness - self.sample_mask
        
        #sample point with the largest weight
        ind = np.unravel_index(np.argmax(sample_heur, axis=None), 
                               sample_heur.shape)
        return tuple(ind)
    
    def _update_from_sample(self, ind):
        """
        Update which is called every time a new sample is taken
        """

        #the reflections of ind along each face of the space/voltage cube
        reflections = [ind]
        ind_list = list(ind)
        for i in range(len(ind_list)):
            ref_ind1 = ind_list.copy()
            ref_ind2 = ind_list.copy()
            ref_ind1[i] = -ind[i]-1
            ref_ind2[i] = 2*self.xyv_dimensions[i] - ind[i] - 1
            reflections.append(tuple(ref_ind1))
            reflections.append(tuple(ref_ind2))
        
        #reflections are considered when measuring relative isolation in order to
        #avoid higher isolations near faces
        for refl in reflections:
            self._update_farness_map(refl)
        
    def _update_farness_map(self,ind):
    	"""
    	Updatae farness_map given a measurement at ind
    	"""

        slices = [slice(2*s-p, 3*s-p) for (s,p) in zip(self.xyv_dimensions,ind)]
        self.farness_map += self.big_distance_map[tuple(slices)]
    
    def _image_reconstruct(self, coords, vals):
        """
        Reconstruct one slice of the image using nearest neighbor regression
        """

        n_neighbors = min(self.n_neigbor_count, len(coords))

        knn = neighbors.KNeighborsRegressor(n_neighbors, 
                                            weights='distance')

        prediction = knn.fit(coords, 
                       vals).predict(self.grid).reshape(
                       self.xyv_dimensions)
        
        for coord, measured_val in zip(coords, vals):
            prediction[tuple(coord)] = measured_val
            
        return prediction

    def _reconstruct(self, num_samples=None):
    	"""
        The reconstruction method used to inform the sampler.
        num_samples should always be the default except for experimental
        purposes
        """

        if num_samples is None:
        	num_samples = len(self.measured_coords)

        coords = self.measured_coords[0:num_samples]
        vals = self.measured_vals[0:num_samples]
        return self._image_reconstruct(coords, vals)
