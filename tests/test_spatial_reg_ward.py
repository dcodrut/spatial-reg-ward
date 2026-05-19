import unittest

import numpy as np

from spatial_reg_ward import SpatialRegWard, make_half_moon_toy_data


class TestSpatialRegWard(unittest.TestCase):
    def test_fit_returns_expected_labels(self):
        data = make_half_moon_toy_data()

        model = SpatialRegWard(
            data["x"],
            data["y"],
            n_clusters=3,
            coords=data["coords"],
            use_cluster_nn=True,
            cluster_nn_k=5,
            min_cluster_size=10,
            pbar=False,
        )
        labels = model.fit()

        self.assertEqual(labels.shape[0], data["x"].shape[0])
        self.assertEqual(len(np.unique(labels)), 3)


if __name__ == "__main__":
    unittest.main()
