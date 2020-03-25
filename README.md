# MapReduce K-means
Parallel implementation of K-means based on MapReduce, using python's multiprocessing API.

### Install
1. Clone this repository
2. ```virtualenv venv && source venv/bin/activate```
3. ```pip install -r requirements.txt```

### Usage
```python
import numpy as np
from models import ParallelKMeans

X, y = np.arange(25).reshape((5, 5)), np.random.randint(2, size=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create a new model instance
model = ParallelKMeans(k=8, n_nodes=10, max_n_iter=10)

# Fit the data, tracking total and iteration elapsed times
total_elapsed, iter_elapsed = model.fit(X_train)

# Predict new labels
predicted_labels = model.predict(X_test)
```

## Experiments
```experiments.py```contains a set of experiments to measure **total/iteration elapsed times** on:
- (1) Different data sizes *N* and number of clusters *k* for a fixed number of parallel nodes *n_nodes=10* (jobs).
- (2) Different number of parallel nodes *n_nodes* (jobs) for for fixed data size *N=100k* points and *k=8* clusters.

(1) and (2) were run on a randomly generated dataset (see ```experiments/gendata_and_result_example.ipynb``` for details). Additionally, (2) was run on the [ULB Machine Learning Group fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) with *N=285k* points (transactions) and *k=2* clusters (fraud/not-fraud).

## Results
Computational performance results can be found in ```experiments/results.ipynb```. An example of the algorithm clustering accuracy on the randomly generated dataset can be found in ```experiments/gendata_and_result_example.ipynb```.
