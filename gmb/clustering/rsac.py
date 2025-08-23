import heapq
from typing import Dict, Set, List

import libpysal
import numpy as np
from scipy.spatial.distance import pdist, squareform


# TODO:
#   - add a min_size parameter to require a minimum number of points in the cluster
#   - add 'complete' and 'average' linkage options
#   - add a verbose flag to print some stats
#   - add documentation
#   - check if the model is fitted before running get_clustering
#   - mention that for the first n points the model will perfectly fit the data
#   - delete the asserts at the end


class RSAC:
    """Regression-based Spatially-constrained Agglomerative Clustering"""

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            *,
            w: libpysal.weights.W = None,
            coords: np.ndarray = None,
            dist_mat: np.ndarray = None,
            use_cluster_knn: bool = False,
            k_extend=1,
            fit_intercept: bool = True,
            dtype=np.float64,
            pbar=True
    ):
        """
        Ward-style clustering with regression cost & spatial constraints.
        """

        if fit_intercept:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        self.x, self.y = x.astype(dtype), y.astype(dtype)
        self.n_samples, self.n_feat = x.shape
        self.dtype = dtype
        self.use_cluster_knn = use_cluster_knn
        self.k_extend = k_extend
        self.pbar = pbar

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y must have the same number of samples, got {x.shape[0]} and {y.shape[0]}.")

        self.coords = np.asarray(coords, dtype=dtype) if coords is not None else None

        # Prepare/check the distance matrix (dist_mat has priority if provided; else compute from coords)
        if dist_mat is not None:
            dm = np.asarray(dist_mat, dtype=self.dtype)
            if dm.shape != (self.n_samples, self.n_samples):
                raise ValueError(f"dist_mat must be shape {(self.n_samples, self.n_samples)}, got {dm.shape}.")

            # Fill diagonal with zeros and check symmetry & non-negativity
            np.fill_diagonal(dm, self.dtype(0.0))
            if not np.allclose(dm, dm.T):
                raise ValueError("dist_mat must be symmetric.")

            if np.any(dm < 0) or np.any(~np.isfinite(dm)):
                raise ValueError("dist_mat must have all finite non-negative values.")

            self.dist_mat = dm
        else:
            if coords is None:
                raise ValueError("Provide either `dist_mat` or `coords` to define spatial distances.")
            self.coords = np.asarray(coords, dtype=self.dtype)
            if self.coords.shape[0] != self.n_samples:
                raise ValueError(
                    f"coords must have n rows == n_samples ({self.n_samples}), got {self.coords.shape[0]}"
                )
            self.dist_mat = squareform(pdist(self.coords)).astype(self.dtype)

        # Infinity constant for convenience
        self._inf = self.dtype(np.inf)

        # DSU parent pointers
        self._parent = {i: i for i in range(self.n_samples)}

        # Init per-cluster stats & member sets
        self._stats, self._members = {}, {}
        for i in range(self.n_samples):
            xi = self.x[i].reshape(-1, 1)
            self._stats[i] = {
                'n_samples': 1,
                'XtX': xi @ xi.T,
                'Xty': (xi.flatten() * self.y[i]),
                'ySS': self.y[i] ** 2,
                'RSS': self.dtype(0.0)
            }
            self._members[i] = {i}

        # 1) Build base adjacency from W (if given, otherwise, we expect use_cluster_knn to be True)
        self.adj_base: Dict[int, Dict[int, np.floating]] = {i: {} for i in range(self.n_samples)}

        if w is not None:
            if w.max_neighbors == 0:
                raise ValueError("The provided weights object `w` has no neighbors. Please check your weights.")
            if len(w.id_order) != self.n_samples:
                raise ValueError(
                    f"The provided weights object `w` has {len(w.id_order)} ids, "
                    f"but we have {self.n_samples} samples. Please check your weights."
                )
            id2pos = {id_: idx for idx, id_ in enumerate(w.id_order)}
            for i_id, nbrs in w.neighbors.items():
                ii = id2pos[i_id]
                for j_id in nbrs:
                    jj = id2pos[j_id]
                    self._add_edge(self.adj_base, ii, jj, self.dist_mat[ii, jj])
        else:
            if not self.use_cluster_knn:
                raise ValueError("No contiguity `w` provided. Either supply `w` or enable `use_cluster_knn=True`.")
        deg = {u: len(nbrs) for u, nbrs in self.adj_base.items()}
        print(
            f"adj_base stats: #nodes: {len(deg)}, #edges: {sum(deg.values()) // 2}, "
            f"min/mean/max degree: {min(deg.values())}/{np.mean(list(deg.values())):.1f}/{max(deg.values())}"
        )

        # 2) If `use_cluster_knn` is set to True, extend the adjacency with k-NN outside each "cluster"
        # (i.e. point since we start with each point as a cluster, so equivalent to a standard k-NN)
        if self.use_cluster_knn:
            if self.k_extend < 1:
                raise ValueError(f"k_extend must be a positive integer if `use_cluster_knn=True', got {self.k_extend}.")

            # Build pre-sorted neighbor index once; store neighbors (excluding self) by ascending distance
            order = np.argsort(self.dist_mat, axis=1).astype(np.int32)
            self._order_mat = np.empty((self.n_samples, self.n_samples - 1), dtype=np.int32)
            for i in range(self.n_samples):
                row = order[i]
                if row[0] == i:
                    self._order_mat[i] = row[1:]
                else:
                    # rare if diagonal not strictly smallest; still drop i
                    self._order_mat[i] = row[row != i][: self.n_samples - 1]
            print("Neighbor index built (pre-sorted rows with cursors).")

            # Per-point advancing cursor (starts at 0, only moves forward)
            self._cursor = np.zeros(self.n_samples, dtype=np.int32)

            # Initialize the extended adjacency structure
            self.adj_cknn: Dict[int, Dict[int, np.floating]] = {i: {} for i in range(self.n_samples)}
            for rep in self._members:
                edges = self._compute_cluster_knn_for(rep)
                for tgt, d in edges.items():
                    self._add_edge(self.adj_cknn, rep, tgt, d)

            deg = {u: len(nbrs) for u, nbrs in self.adj_cknn.items()}
            print(
                f"adj_cknn stats: #nodes: {len(deg)}, #edges: {sum(deg.values()) // 2}, "
                f"min/mean/max degree: {min(deg.values())}/{np.mean(list(deg.values())):.1f}/{max(deg.values())}"
            )

        # merge the base and extended adjacency into a single adjacency structure
        self._refresh_adj()

        # Initialize the _heap using tuples of (delta_RSS, spatial_dist, u, v, n, RSS)
        # => the first two values are used to sort the _heap, the rest is used for merging
        self._heap: List[tuple] = []
        seen = set()
        for u, nbrs in self.adj.items():
            for v, dist in nbrs.items():
                if (u, v) in seen or (v, u) in seen:
                    continue
                seen.add((u, v))
                rss, delta_rss = self._delta_rss(u, v)
                heapq.heappush(self._heap, (delta_rss, dist, u, v, 2, rss))  # n=2 for the first merge
        heapq.heapify(self._heap)
        print(f"Built initial heap with {len(self._heap)} candidate merges.")

        # Record all the intermediate clustering states so we can retrieve them later
        self.history = {self.n_samples: {i: i for i in range(self.n_samples)}}

    def _add_edge(self, layer: Dict[int, Dict[int, np.floating]], u: int, v: int, d):
        """Undirected min-weight add: ensure u<->v exist with the smaller of existing and d."""
        if u == v:
            return

        if v in layer[u] and u in layer[v]:
            assert layer[u][v] == layer[v][u], "Asymmetric edge weight detected"

        if u not in layer:
            layer[u] = {}

        if v not in layer:
            layer[v] = {}

        # Get the minimum distance and set it both ways
        best = min(d, layer[u].get(v, self._inf))
        layer[u][v] = layer[v][u] = best

    def _nearest_outside_neighbor(self, p: int, rep: int):
        """Return (target_rep, distance) for the nearest neighbor of point p that is **outside** cluster `rep`.
        Uses the pre-sorted neighbor list with a per-point cursor that only advances (lazy deletion).
        """
        row = self._order_mat[p]
        i = int(self._cursor[p])  # current pointer into row
        m = row.shape[0]

        # Advance past same-cluster neighbors lazily
        while i < m and self._find(int(row[i])) == rep:
            i += 1
        self._cursor[p] = i  # persist new pointer (monotone increasing)
        if i < m:
            q = int(row[i])
            return self._find(q), self.dist_mat[p, q]
        return None

    def _compute_cluster_knn_for(self, rep: int) -> Dict[int, np.floating]:
        """Compute (but do not apply) up to k_extend target clusters and distances for `rep`.

        We look ahead in each member's neighbor list and gather the best distinct targets until we have gathered
        `k_extend` distinct targets or exhausted all members.

        This scan does not consume neighbors globally, except that we persist skipping of now-inside-cluster neighbors
        by advancing the stored cursor up to the first outside neighbor.
        """
        best_per_target: Dict[int, np.floating] = {}

        for p in self._members[rep]:
            row = self._order_mat[p]
            i = int(self._cursor[p])
            m = row.shape[0]

            # Advance to first outside-neighbor for current membership and persist that position
            while i < m and self._find(int(row[i])) == rep:
                i += 1
            self._cursor[p] = i

            # Look ahead locally to gather more distinct targets, without consuming the global cursor
            j = i
            while j < m and len(best_per_target) < self.k_extend:
                q = int(row[j])
                tgt = self._find(q)
                if tgt != rep:
                    d = self.dist_mat[p, q]
                    prev = best_per_target.get(tgt)
                    if prev is None or d < prev:
                        best_per_target[tgt] = d
                j += 1

            if len(best_per_target) >= self.k_extend:
                break

        # Top-k distinct target clusters by min witness distance
        out: Dict[int, np.floating] = {}
        for tgt, d in sorted(best_per_target.items(), key=lambda t: t[1])[: self.k_extend]:
            out[tgt] = d
        return out

    def _refresh_adj(self):
        """Update the union of base and cKNN adjacency."""

        if not self.use_cluster_knn:
            # If we don't extend adjacency, just use the base adjacency
            self.adj = self.adj_base
            return

        adj: Dict[int, Dict[int, np.floating]] = {i: {} for i in range(self.n_samples)}

        for u, nbrs in self.adj_base.items():
            for v, d in nbrs.items():
                self._add_edge(adj, u, v, d)

        for u, nbrs in self.adj_cknn.items():
            for v, d in nbrs.items():
                self._add_edge(adj, u, v, d)

        self.adj = adj

    def _find(self, u):
        """Path-compressed DSU find."""
        if self._parent[u] != u:
            self._parent[u] = self._find(self._parent[u])
        return self._parent[u]

    @staticmethod
    def _safe_solve(xtx, xty):
        """ Solve linear regression coefficients safely. Try to use direct solve, fallback to pseudo-inverse. """
        try:
            return np.linalg.solve(xtx, xty)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(xtx) @ xty

    def _delta_rss(self, u, v):
        ru, rv = self._find(u), self._find(v)
        sa, sb = self._stats[ru], self._stats[rv]
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

    def _rebuild_base_after_merge(self, keep: int, drop: int):
        """Contract base adjacency after merging `drop` into `keep`.

        For single-linkage, d(A + B, C) = min(d(A, C), d(B, C)).
        We therefore update weights without any pointwise distance computations.
        """

        # Capture old neighbor maps
        nbrs_keep = dict(self.adj_base.get(keep, {}))
        nbrs_drop = dict(self.adj_base.get(drop, {}))
        all_nbrs = (set(nbrs_keep.keys()) | set(nbrs_drop.keys())) - {keep, drop}

        # Remove old references to keep/drop from neighbors
        for nb in nbrs_keep:
            self.adj_base[nb].pop(keep, None)
        for nb in nbrs_drop:
            self.adj_base[nb].pop(drop, None)

        # New neighbor map for keep
        new_map = {}
        for nb in all_nbrs:
            d1 = nbrs_keep.get(nb, self._inf)
            d2 = nbrs_drop.get(nb, self._inf)
            d = d1 if d1 < d2 else d2
            if np.isfinite(d):
                new_map[nb] = d
        self.adj_base[keep] = {}
        for nb, d in new_map.items():
            self._add_edge(self.adj_base, keep, nb, d)

        # Remove representative `drop`
        self.adj_base.pop(drop, None)

    def _rebuild_cknn_after_merge_for(self, reps: Set[int]):
        """
        Two-steps local rebuild for cKNN on a set of representatives:
            Step 1: compute new edges for each rep.
            Step 2: wipe current incident edges for those reps and re-add exactly the new edges (symmetrically).
        """
        if not self.use_cluster_knn or not reps:
            return

        # Step 1: compute new edges per rep
        new_edges_map: Dict[int, Dict[int, np.floating]] = {}
        for rep in reps:
            if rep in self._members:  # still alive
                new_edges_map[rep] = self._compute_cluster_knn_for(rep)

        # Step 2: wipe current incident edges for touched reps (symmetrically)
        for rep in new_edges_map.keys():
            for nb in list(self.adj_cknn.get(rep, {}).keys()):
                # remove both directions
                self.adj_cknn[nb].pop(rep, None)
            self.adj_cknn[rep] = {}

        # Re-add the new edges
        for rep, edges in new_edges_map.items():
            for nb, d in edges.items():
                self._add_edge(self.adj_cknn, rep, nb, d)

    def _merge(self, u, v, new_rss):
        # Get the root representatives of u and v
        ru, rv = self._find(u), self._find(v)

        # They should not be the same
        assert ru != rv, f"The representatives of {u} and {v} are the same: {ru} == {rv}"

        # Set the new representative
        keep, drop = ru, rv

        # Save the cKNN neighbors before the merge (for touched set)
        if self.use_cluster_knn:
            cknn_neigh_keep = set(self.adj_cknn.get(keep, {}).keys())
            cknn_neigh_drop = set(self.adj_cknn.get(drop, {}).keys())

        # Update the base adjacency structure after merging
        self._rebuild_base_after_merge(keep, drop)

        # Merge the members of the two clusters
        self._parent[drop] = keep
        self._members[keep] |= self._members[drop]
        del self._members[drop]

        # Merge the stats into the new representative and delete the old ones
        sa, sb = self._stats[keep], self._stats[drop]
        sa['n_samples'] += sb['n_samples']
        sa['XtX'] += sb['XtX']
        sa['Xty'] += sb['Xty']
        sa['ySS'] += sb['ySS']
        sa['RSS'] = new_rss  # use the provided new_rss for the merged cluster which was kept in the _heap
        del self._stats[drop]

        # Update cluster‑kNN layer (with local rebuild, i.e. only the affected representatives)
        if self.use_cluster_knn:
            # These are the nodes whose cluster‑kNN needs to be recomputed
            touched: Set[int] = {keep} | cknn_neigh_keep | cknn_neigh_drop

            # Remove the dropped rep’s entry and all edges to it
            for nb in list(self.adj_cknn.get(drop, {}).keys()):
                self.adj_cknn[nb].pop(drop, None)
            self.adj_cknn.pop(drop, None)

            # Recompute cluster‑kNN for all the touched nodes
            self._rebuild_cknn_after_merge_for(touched)

        # Refresh the adjacency structure
        self._refresh_adj()

        return keep

    def fit(self):
        """
        Run until one cluster remains, saving history at each step.
        """

        if self.pbar:
            from tqdm import tqdm
            pbar = tqdm(total=self.n_samples - 1, desc='RSAC merges')

            # Compute TSS for the entire dataset (for information only)
            y_mean = np.mean(self.y)
            tss = np.sum((self.y - y_mean) ** 2)

        k = self.n_samples  # current number of clusters
        while k > 1 and self._heap:
            delta_rss, dist, u, v, n, rss = heapq.heappop(self._heap)
            ru, rv = self._find(u), self._find(v)

            # If they are already in the same cluster, skip this pair
            if ru == rv:
                continue

            # Secondly, it could be that the clusters of u and v grew meanwhile => delta_RSS is obsolete
            # We check how many samples we had at the time when we computed the delta_RSS and then compare it to the
            # actual size of the cluster we are about to build
            if n != (self._stats[ru]['n_samples'] + self._stats[rv]['n_samples']):  # obsolete merge, skip it
                continue

            # Merge the two clusters
            new_rep = self._merge(ru, rv, rss)
            k -= 1

            # Save the current configuration
            self.history[k] = {i: self._find(i) for i in range(self.n_samples)}

            # Push fresh candidates between the new cluster and its neighbors
            for nbr, d_curr in list(self.adj[new_rep].items()):
                r_nbr = self._find(nbr)
                if r_nbr == new_rep:
                    continue
                rss_new, delta_new = self._delta_rss(new_rep, r_nbr)

                # Use current union adjacency weight for tie-breaking
                dist_new = self.adj[new_rep].get(r_nbr, self._inf)
                n_new = self._stats[new_rep]['n_samples'] + self._stats[r_nbr]['n_samples']
                heapq.heappush(self._heap, (delta_new, dist_new, new_rep, r_nbr, n_new, rss_new))

            if self.pbar:
                # Compute the total RSS over all clusters (for information only) and the resulting R^2
                total_rss = sum(s['RSS'] for s in self._stats.values())
                r2 = 1.0 - max(total_rss, 0.0) / tss if tss > 0 else 1.0
                pbar.set_postfix({'k': k, 'R²': f"{r2:.4f}"})
                pbar.update(1)

        # Clean up the _heap
        self._heap.clear()

        if k != 1:
            print(f"Warning: stopped with {k} clusters remaining (probably disconnected components).")

        return self

    def get_labels(self, k):
        """Retrieve labels for exactly k clusters."""
        return self.history.get(k)
