# -*- coding: utf-8 -*-

# Third-party dependencies
from multiprocessing import Pool
from itertools import groupby
from operator import itemgetter
import numpy as np
import time
import json

__author__ = "Antonio Javier Samaniego Jurado"
__email__ = "samaniegoads@gmail.com"
__version__ = "1.0.0"

class ParallelKMeans():
	"""
    A class to run a MapReduce-based parallel version of k-means.

    Attributes
    ----------
    k : int
        number of clusters
    n_nodes : int
        number of processes to distribute the algorithm on
    max_n_iter : int
        maximum number of iterations for the algorithm to run (convergence constraint)
    mean_change_coef : float
        what proportion of the means change in distance from iteration 1 to 2, i.e. d(M1,M2),
        is taken as the minimum threshold Dmin. Converge is met at iteration i if d(Mi,Mi+1) < Dmin
	means : dict
		current means/centroids

    Methods
    -------
    get_params()
        returns tuple with class instance attributes
	
	get_means()
        returns current means/centroids

	mapper(x)
		performs the map step, returning a key-value tuple in the form of (assigned_cluster, point)

	reducer(mapped_data)
		performs the reduce step, returning a dict with the new means

	gen_init_centroids(data)
		randomly generates initial centroids based on the input data, returning them as a dict
	
	means_check(new_means, min_dist_threshold)
		performs convergence check based on last iteration means/centroids change 

	fit(data)
		runs the algorithm on the input data, returning total/iteration elapsed times

	predict(data)
		returns the cluster labels for new, unseen data
    """

	# Default attributes
	k = 8
	n_nodes = 10
	max_n_iter = 10
	mean_change_coef = 0.05
	means = {}


	def __init__(self, k, n_nodes, max_n_iter, mean_change_coef=0.05):
		"""
        Parameters
        ----------
        k : int
	        number of clusters
	    n_nodes : int
	        number of processes to distribute the algorithm on
	    max_n_iter : int
	        maximum number of iterations for the algorithm to run (convergence constraint)
	    mean_change_coef : float
	        what proportion of the means change in distance from iteration 1 to 2, i.e. d(M1,M2),
	        is taken as the minimum threshold Dmin. Converge is met at iteration i if d(Mi,Mi+1) < Dmin
	        (default=0.05)
        """

		self.k = k 
		self.n_nodes = n_nodes
		self.max_n_iter = max_n_iter 
		self.mean_change_coef = mean_change_coef
	    

	def get_params(self):
		"""
		Getter for instance attributes

        Returns
        ------
        instance attributes : tuple
        """
		return (self.k, self.n_nodes, self.max_n_iter, self.mean_change_coef)


	def get_means(self):
		"""
		Getter for current means/centroids
		
        Returns
        ------
        current means/centroids : dict
        """
		return self.means


	def mapper(self, x):
		"""
		Performs the map step, executing on each parallel node (job) one point at a time.
		Measures the euclidean distance between the input point x and each current mean/centroid,
		assigning it to the closest.

		Parameters
        ----------
        x : array-like
            Data point to perform the cluster assignment on.
		
        Returns
        ------
        key-value tuple in the form of (assigned_cluster, point) : tuple
        """
		distances = {}
		for key, m in self.means.items():
			distances[key] = np.linalg.norm(x - m)
		return (min(distances, key=distances.get), x)


	def reducer(self, mapped_data):
		"""
		Performs the reduce step. Takes the list of mapped tuples, groups the points (value) by cluster (key)
		and recomputes each corresponding cluster mean/centroid.

		Parameters
        ----------
        mapped_data : list
            List of mapped tuples
		
        Returns
        ------
        Recomputed means/centroids for each cluster : dict
        """  
		new_means = []
		for key, item in groupby(sorted(mapped_data, key=itemgetter(0)), itemgetter(0)):
			points = list(map(itemgetter(1), item))
			new_means.append((key, sum(points) / len(points)))
		return new_means


	def gen_init_centroids(self, data):
		"""
		Randomly generates initial centroids. It randomly chooses k centroids within the 
		range of each input dimension. For instance, if data = array([1, 11], [10, 20]),
		the initial centroids would be random 2-D points between 1 and 10 on the first dimension
		and between 11 and 20 on the second dimension.

		Parameters
        ----------
        data : array-like
            Input data to run the algorithm on 
		
        Returns
        ------
        Initial means/centroids for each cluster : dict
        """ 
		initial_centroids = {}
		dimensions = data.shape[1]

		lower_bound = np.amin(data, axis=0)  
		upper_bound = np.amax(data, axis=0)

		random_points = np.c_[np.random.randint(lower_bound[0], upper_bound[0], (self.k))]
		for i in range(0, dimensions - 1):
			new_points_dimension = np.c_[np.random.randint(lower_bound[i], upper_bound[i], (self.k))]
			random_points = np.concatenate((random_points, new_points_dimension), axis=1)

		for idx,v in enumerate(random_points):
			initial_centroids[str(idx + 1)] = v
		
		return initial_centroids


	def means_check(self, new_means, min_dist_threshold):
		"""
		Perfoms convergence check based on the last iteration means/centroids change. 
		Taking a minimum threshold Dmin, convergence is met at iteration i if d(Mi,Mi+1) < Dmin

		Parameters
        ----------
        new_means : dict
        	means/centroids of iteration i
        min_dist_threshold : float
	        minimum threshold Dmin.
		
        Returns
        ------
        True if convergence is met, false otherwise : bool
        """ 
		means_dist_change = np.linalg.norm(np.array(list(self.means.values())) - np.array(list(new_means.values())))
		if  means_dist_change < min_dist_threshold:
			return True
		return False


	def fit(self, data):
		"""
		Runs the algorithm on the input data, returning total/iteration elapsed times.

		Parameters
        ----------
        data : array-like
        	Input data to fit the model on 

        Returns
        ------
        Total/iteration elapsed times : tuple
        """ 

		# Create a new multiprocessing Pool object to map and process data across n_nodes
		pool = Pool(self.n_nodes)

		# Generate random initial centroids
		initial_centroids = self.gen_init_centroids(data)

		# Run max_n_iter
		self.means = initial_centroids

		iter_elapsed_times = []
		start_total = time.time()
		for i in range(1, self.max_n_iter + 1):

			start_iter = time.time()

			# Map data
			mapped_data = pool.map(self.mapper, data)

			# Reduce data
			reduced = self.reducer(mapped_data)

			# Convergence check for early stopping
			# Copy means dict to avoid undesirable update() mutations
			means_copy = self.means
			means_copy.update(dict(reduced))

			# Get maximum means change in 1st iteration for min_dist_threshold
			if i == 1:
				max_means_change = np.linalg.norm(np.array(list(self.means.values())) - np.array(list(means_copy.values())))
				min_dist_threshold = self.mean_change_coef * max_means_change

			stop_means = self.means_check(means_copy, min_dist_threshold)


			# Update means dict with new, reduced values
			self.means.update(dict(reduced))

			end_iter = time.time()

			iter_time = round(end_iter - start_iter, 5)
			iter_elapsed_times.append(iter_time)
			print("Iter", i, "took", iter_time, "seconds.")

			if stop_means:
				print("Early stopping (means didn't change).")
				break

		end_total = time.time()
		total_elapsed = round(end_total - start_total, 5)
		print("Elapsed time", total_elapsed, "seconds.")

		return (total_elapsed, iter_elapsed_times)


	def predict(self, data):
		"""
		Predicts cluster labels for new, unseen data.

		Parameters
        ----------
        data : array-like
        	Input data to predict on 

        Returns
        ------
        predicted : dict
			Dict with each cluster (key) and assigned list of points (values)
        """ 

		# Map data
		mapped_data = pool.map(self.mapper, data)

		# Reduce by just grouping points assigned to each cluster (without recomputing the mean, as this method only predicts)
		predicted = {}
		for key, item in groupby(sorted(mapped_data, key=itemgetter(0)), itemgetter(0)):
			points = list(map(itemgetter(1), item))
			predicted[key] = points
		return predicted


