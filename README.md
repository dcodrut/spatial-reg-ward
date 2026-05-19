Spatial-reg-ward
=================

Python package providing `SpatialRegWard`, a spatially constrained agglomerative clustering
algorithm that uses a regression-based Ward-style objective.

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

Development
-----------

Install requirements:

```bash
pip install -r requirements.txt
```

