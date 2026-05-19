Spatial-reg-ward
=================

Python package providing `SpatialRegWard`, a spatially constrained agglomerative clustering
algorithm that uses a regression-based Ward-style objective.

Install from GitHub
------------------

```bash
pip install git+https://github.com/dcodrut/spatial-reg-ward.git
```


Quick usage example
-------------------

```python
import numpy as np
from spatial_reg_ward import SpatialRegWard

# X: features (n_samples, n_features)
# y: response (n_samples,)
# coords: spatial coordinates (n_samples, 2)

# model = SpatialRegWard(X, y, n_clusters=5, coords=coords)
# labels = model.fit()
```

Toy dataset helper
-----------------

```python
from spatial_reg_ward import make_half_moon_toy_data

data = make_half_moon_toy_data()
```

Example script
--------------

```bash
python examples/run_example.py
```

Testing
-------

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Development
-----------

Install requirements:

```bash
pip install -r requirements.txt
```
