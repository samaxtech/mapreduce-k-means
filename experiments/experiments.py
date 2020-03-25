# -*- coding: utf-8 -*-

# Dependencies
# Local
from models import ParallelKMeans

# Third-party
import pandas as pd
import numpy as np
import json
import sys

__author__ = "Antonio Javier Samaniego Jurado"
__email__ = "samaniegoads@gmail.com"
__version__ = "1.0.0"

def generate_data(N):
	"""
	Randomly generates data points between 0 and 1 based on a normal distribution.
	The generated data tends to be groupped in 3 clusters by default.

    Parameters
    ----------
    N : int
        Data size (# points)

 	Returns
    ------
    data : array-like
    	randomly generated data 
    """
	c1 = np.random.normal(loc=[0, 1], scale=[0.5, 0.75], size=(N, 2))
	c2 = np.random.normal(loc=[2, 2.2], scale=[0.5, 0.75], size=(N, 2))
	c3 = np.random.normal(loc=[5, 7], scale=[0.25, 0.8], size=(N, 2))
	data = np.concatenate((c1, c2, c3), axis=0)
	return data


def run_experiment(exp_id):
	##################################################################################
	# Experiments: 1 (generated dataset), 2 (fraud dataset)
	# Assumptions in all experiments: Fixed max_n_iter=10 and mean_change_coef=0.05
	##################################################################################

	# ------------------------------------------------------------
	# Experiment 1.1: Test different N and k for fixed n_nodes=10
	# ------------------------------------------------------------
	if exp_id == 'gen1':
		print("-------------------------------------------------------------------------------")
		print("Experiment 1.1 (generated dataset): Test different N and k for fixed n_nodes=10")
		print("-------------------------------------------------------------------------------")
		n_nodes = 10
		k_clusters = [2, 5, 10]
		N_sizes = [10000, 100000, 1000000]

		# Elapsed times
		elapsed_times = {}

		for N in N_sizes:
			# Generate random cluster points
			data = generate_data(N)
			
			for k in k_clusters:
				# Fit the model
				model = ParallelKMeans(k=k, n_nodes=10, max_n_iter=10)

				print("\nRunning for", N, "points with", k, "clusters.")
				total_elapsed, iter_elapsed = model.fit(data)
				elapsed_times[str(N) + '_' + str(k)] = {'total': total_elapsed, 'iter': iter_elapsed}

		# Save elapse_times into JSON
		out = 'out/generated_dataset/elapsed_times_e1.json'
		with open(out, 'w') as fp:
		    json.dump(elapsed_times, fp)
		    print("Elapsed times saved at", out)

	# ----------------------------------------------------------
	# Experiment 1.2: Test different n_nodes for fixed N and k
	# ----------------------------------------------------------
	elif exp_id == 'gen2':
		print("-----------------------------------------------------------------------------------")
		print("Experiment 1.2 (generated dataset): Test different n_nodes for fixed N=100k and k=8")
		print("-----------------------------------------------------------------------------------")
		n_nodes = [1, 2, 4, 8, 12, 16]
		k_clusters = 8
		N = 100000

		# Elapsed times
		elapsed_times = {}

		# Generate random cluster points
		data = generate_data(N)

		for n in n_nodes:
			# Fit the model
			model = ParallelKMeans(k=k_clusters, n_nodes=n, max_n_iter=10)

			print("\nRunning for", n, "nodes.")
			total_elapsed, iter_elapsed = model.fit(data)
			elapsed_times[str(n)] = {'total': total_elapsed, 'iter': iter_elapsed}

		# Save elapse_times into JSON
		out = 'out/generated_dataset/elapsed_times_e2.json'
		with open(out, 'w') as fp:
		    json.dump(elapsed_times, fp)
		    print("Elapsed times saved at", out)


	# ----------------------------------------------------------
	# Experiment 2: Test different n_nodes for fixed N and k
	# ----------------------------------------------------------
	elif exp_id == 'fraud':
		print("-------------------------------------------------------------------------------")
		print("Experiment 2.2 (fraud dataset): Test different n_nodes for fixed N=285k and k=2")
		print("-------------------------------------------------------------------------------")
		n_nodes = [1, 2, 4, 8, 12, 16]
		k_clusters = 2

		# Elapsed times and final means
		elapsed_times = {}
		means = {}

		# Read in fraud data, remove target col (fraud/not-fraud) and convert to numpy array
		fraud_df = pd.read_csv('../data/creditcard.csv')
		fraud_df = fraud_df.loc[:, fraud_df.columns != 'Class']
		fraud_df = np.array(fraud_df)

		for n in n_nodes:
			# Fit the model
			model = ParallelKMeans(k=k_clusters, n_nodes=n, max_n_iter=10)

			print("\nRunning for", n, "nodes.")
			total_elapsed, iter_elapsed = model.fit(fraud_df)
			elapsed_times[str(n)] = {'total': total_elapsed, 'iter': iter_elapsed}

		# Save elapse_times and means into JSON
		out = 'out/fraud_dataset/elapsed_times_fraud.json'
		with open(out, 'w') as fp:
			json.dump(elapsed_times, fp)
			print("Elapsed times saved at", out)


	else:
		print("Please specify a valid experiment id.")



if __name__ == '__main__':
	exp_id = sys.argv[1]
	run_experiment(exp_id)











