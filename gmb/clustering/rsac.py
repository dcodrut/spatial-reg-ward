import heapq

import numpy as np
from haversine import haversine_vector
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


# TODO:
#   1. add a progress bar
#   2. add a min_size parameter to require a minimum number of points in the cluster
#   3. add a dynamic adjacency mode that uses k-NN outside of the cluster
#   4. add a static adjacency mode that uses a user-provided adjacency matrix
#   5. add 'complete' and 'average' linkage options
#   6. add a verbose flag to print some stats
#   7. compute the R² score also
#   8. add documentation
#   9. check if the model is fitted before running get_clustering
#   10. mention that for the first n points the model will perfectly fit the data
#   11. parametrize kNN and MST


class RSAC:
    """Regression-based Spatially-constrained Agglomerative Clustering"""

    def __init__(self,
                 x, y, coords,
                 fit_intercept=True,
                 dynamic_adjacency=False,
                 static_adjacency=None,
                 knn_k=3,
                 distance_metric='euclidean',
                 dtype=np.float64):
        """
        Ward-style clustering with regression cost & spatial constraints.
        """
        if fit_intercept:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        self.x, self.y = x.astype(dtype), y.astype(dtype)
        self.n_samples, self.n_feat = x.shape
        self.dtype = dtype
        self.coords = np.asarray(coords)
        self.dynamic_adjacency = dynamic_adjacency
        self.static_adjacency = static_adjacency
        self.knn_k = knn_k

        # DSU parent pointers
        self.parent = {i: i for i in range(self.n_samples)}

        # Init per-cluster stats & member sets
        self.stats, self.members = {}, {}
        for i in range(self.n_samples):
            xi = x[i].reshape(-1, 1)
            self.stats[i] = {
                'n_samples': 1,
                'XtX': xi @ xi.T,
                'Xty': (xi.flatten() * y[i]),
                'ySS': y[i] ** 2,
                'RSS': dtype(0.0)
            }
            self.members[i] = {i}

        # Precompute the distance matrix depending on the metric
        if distance_metric == 'euclidean':
            self.dist_mat = squareform(pdist(self.coords)).astype(self.dtype)
        elif distance_metric == 'haversine':
            self.dist_mat = haversine_vector(self.coords, self.coords).astype(self.dtype)
        else:
            raise ValueError(f"Unknown distance_metric {distance_metric}")
        self._dist = lambda i, j: self.dist_mat[i, j]  # direct access function

        # Build the adjacency structure depending on the mode
        if self.dynamic_adjacency:
            self._build_dynamic_adjacency_all()
        else:
            if self.static_adjacency is not None:
                self._use_external_static_adjacency()
            else:
                self._build_static_adjacency()

        # Initialize the heap using tuples of (delta_RSS, spatial_dist, u, v, n, RSS)
        # => the first two values are used to sort the heap, the rest is used for merging
        self.heap = []
        seen = set()
        for u, nbrs in self.adj.items():
            for v, dist in nbrs.items():
                if (u, v) in seen or (v, u) in seen:
                    continue
                seen.add((u, v))
                rss, delta_rss = self._delta_rss(u, v)
                heapq.heappush(self.heap, (delta_rss, dist, u, v, 2, delta_rss))  # n=2 for the first merge

        # Record all the intermediate clustering states so we can retrieve them later
        self.history = {self.n_samples: {i: i for i in range(self.n_samples)}}

    def _use_external_static_adjacency(self):
        """Convert static_adjacency input into self.adj."""
        adj = {i: {} for i in range(self.n_samples)}
        sa = self.static_adjacency
        if isinstance(sa, dict):
            for u, nbrs in sa.items():
                if isinstance(nbrs, dict):
                    for v, d in nbrs.items():
                        adj[u][v] = d
                        adj[v][u] = d
                else:
                    for v in nbrs:
                        d = self._dist(u, v)
                        adj[u][v] = d
                        adj[v][u] = d
        else:
            for u, v in sa:
                d = self._dist(u, v)
                adj[u][v] = d
                adj[v][u] = d
        self.adj = adj

    def _build_static_adjacency(self):
        """k-NN + MST based on precomputed dist_mat."""
        self.adj = {i: {} for i in range(self.n_samples)}

        # k-NN
        for i in range(self.n_samples):
            order = np.argsort(self.dist_mat[i])
            for j in order[1: self.knn_k + 1]:
                d = self._dist(i, j)
                self.adj[i][int(j)] = d
                self.adj[int(j)][i] = d

        # count the number of edges
        n_edges = sum(len(nbrs) for nbrs in self.adj.values()) // 2
        print(f"Number of edges in the k-NN graph: {n_edges}")

        # MST
        mst = minimum_spanning_tree(self.dist_mat).toarray().astype(self.dtype)
        rows, cols = np.nonzero(mst)
        for i, j in zip(rows, cols):
            if i < j:
                self.adj[int(i)][int(j)] = mst[i, j]
                self.adj[int(j)][int(i)] = mst[i, j]

        n_edges = sum(len(nbrs) for nbrs in self.adj.values()) // 2
        print(f"Number of edges after adding the MST graph: {n_edges}")

    def _find(self, u):
        """Path-compressed DSU find."""
        if self.parent[u] != u:
            self.parent[u] = self._find(self.parent[u])
        return self.parent[u]

    def _build_dynamic_adjacency_all(self):
        """Initial per-point NN-outside using dist_mat."""
        self.adj = {i: {} for i in range(self.n_samples)}
        for i in range(self.n_samples):
            dists = self.dist_mat[i].copy()
            dists[i] = np.inf
            j = np.argmin(dists)
            if not np.isinf(dists[j]):
                d = self.dtype(dists[j])
                ri, rj = self._find(i), self._find(j)
                self.adj[ri][rj] = d
                self.adj[rj][ri] = d
        n_edges = sum(len(nbrs) for nbrs in self.adj.values()) // 2
        print(f"Number of edges in the initial dynamic adjacency: {n_edges}")

    @staticmethod
    def _safe_solve(xtx, xty):
        """ Solve linear regression coefficients safely. Try to use direct solve, fallback to pseudo-inverse. """
        try:
            return np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(xtx) @ xty

    def _delta_rss(self, u, v):
        ru, rv = self._find(u), self._find(v)
        sa, sb = self.stats[ru], self.stats[rv]
        n_samples_total = sa['n_samples'] + sb['n_samples']

        # Return 0 if we don't have enough points to fit a linear model (as we will perfectly fit the data)
        if n_samples_total <= self.n_feat:
            return self.dtype(0.0), self.dtype(0.0)

        # Get the RSS for the merged cluster
        xtx = sa['XtX'] + sb['XtX']
        xty = sa['Xty'] + sb['Xty']
        beta = self._safe_solve(xtx, xty)
        yss = sa['ySS'] + sb['ySS']
        rss_ab = yss - beta @ xty
        delta_rss = rss_ab - (sa['RSS'] + sb['RSS'])

        # Merging two clusters should always increase the RSS (if not, probably due to the pseudo-inverse fallback)
        if delta_rss < 0:
            print(
                f"Warning: negative delta_RSS for clusters {ru} and {rv} (n_samples_total = {n_samples_total})"
                f": {delta_rss:.3f} < 0. Setting to 0."
            )
            delta_rss = self.dtype(0.0)
            rss_ab = sa['RSS'] + sb['RSS']

        return rss_ab, delta_rss

    def _spatial_dist(self, u, v):
        """
        Min-distance between clusters u,v using dist_mat (~single-linkage distance for the clusters).
        """
        pu = list(self.members[u])
        pv = list(self.members[v])
        dists = self.dist_mat[np.ix_(pu, pv)]
        return np.min(dists)

    def _merge(self, u, v, new_rss):
        # Get the root representatives of u and v
        ru, rv = self._find(u), self._find(v)

        # They should not be the same
        assert ru != rv, f"The representatives of {u} and {v} are the same: {ru} == {rv}"

        # 2) Set the new representative
        new_rep, old_rep = ru, rv

        # 3) Contract the neighborhoods of the two clusters
        nbrs_new = set(self.adj.get(new_rep, {}).keys())
        nbrs_old = set(self.adj.get(old_rep, {}).keys())
        nbrs_combined = (nbrs_new | nbrs_old) - {new_rep, old_rep}

        # 3.1) Remove old edges between the representatives of the two clusters and their neighbors
        for nbr in nbrs_new:
            self.adj[nbr].pop(new_rep, None)
        for nbr in nbrs_old:
            self.adj[nbr].pop(old_rep, None)

        # 3.2) Build contracted adjacency for new_rep
        self.adj[new_rep] = {}
        for nbr in nbrs_combined:
            d = self._dist(new_rep, nbr)
            self.adj[new_rep][nbr] = d
            self.adj[nbr][new_rep] = d

        # 3.3) Remove old_rep from adjacency dict
        if old_rep in self.adj:
            del self.adj[old_rep]

        # 4) Merge the members of the two clusters
        self.parent[old_rep] = new_rep
        self.members[new_rep] |= self.members[old_rep]
        del self.members[old_rep]

        # 5) Merge the stats into the new representative and delete the old ones
        sa, sb = self.stats[new_rep], self.stats[old_rep]
        sa['n_samples'] += sb['n_samples']
        sa['XtX'] += sb['XtX']
        sa['Xty'] += sb['Xty']
        sa['ySS'] += sb['ySS']
        sa['RSS'] = new_rss  # use the provided new_rss for the merged cluster which was kept in the heap
        del self.stats[old_rep]

        # If used, update the adjacency structure
        # (clean up the old edges & add new edges between the new representative and its external closest neighbors)
        if self.dynamic_adjacency:
            for nbr in list(self.adj[new_rep]):
                self.adj[nbr].pop(new_rep, None)
            self.adj[new_rep].clear()
            for pt in self.members[new_rep]:
                dists = self.dist_mat[pt].copy()
                for j in self.members[new_rep]:
                    dists[j] = self.dtype(np.inf)
                jmin = np.argmin(dists)
                if not np.isinf(dists[jmin]):
                    rj = self._find(jmin)
                    if rj != new_rep:
                        d = dists[jmin]
                        self.adj[new_rep][rj] = d
                        self.adj[rj][new_rep] = d
        return new_rep

    def fit(self):
        """
        Run until one cluster remains, saving history at each step.
        """
        heapq.heapify(self.heap)
        print(f"Initial heap size: {len(self.heap)}")

        k = self.n_samples  # current number of clusters
        while k > 1 and self.heap:
            delta_rss, dist, u, v, n, rss = heapq.heappop(self.heap)
            ru, rv = self._find(u), self._find(v)

            # If they are already in the same cluster, skip this pair
            if ru == rv:
                continue

            # Secondly, it could be that the clusters of u and v grew meanwhile => delta_RSS is obsolete
            # We check how many samples we had at the time when we computed the delta_RSS and then compare it to the
            # actual size of the cluster we are about to build
            if n != (self.stats[ru]['n_samples'] + self.stats[rv]['n_samples']):  # obsolete merge, skip it
                continue

            # Merge the two clusters
            new_rep = self._merge(ru, rv, rss)
            k -= 1

            # Save the current configuration
            self.history[k] = {i: self._find(i) for i in range(self.n_samples)}

            # Add new candidate merges between the newly formed cluster and its neighbors
            for nbr, _ in self.adj[new_rep].items():
                rss_new, delta_rss_new = self._delta_rss(new_rep, nbr)
                dist_new = self._spatial_dist(new_rep, nbr)
                n_samples_new = self.stats[new_rep]['n_samples'] + self.stats[nbr]['n_samples']
                heapq.heappush(self.heap, (delta_rss_new, dist_new, new_rep, nbr, n_samples_new, rss_new))

        # Clean up the heap
        self.heap.clear()

        return self

    def get_labels(self, k):
        """Retrieve labels for exactly k clusters."""
        return self.history.get(k)
