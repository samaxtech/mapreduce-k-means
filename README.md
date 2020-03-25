# MapReduce K-means
Parallel implementation of K-means based on MapReduce using python's multiprocessing API.

### Install
1. Clone this repository
2. ```virtualenv venv && source venv/bin/activate```
2. ```pip install -r requirements.txt```

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
- Generated: Random 
