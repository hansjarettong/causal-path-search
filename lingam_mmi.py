import pandas as pd
import numpy as np
from typing import List
from scipy.special import digamma
from lingam.base import _BaseLiNGAM
from lingam.direct_lingam import DirectLiNGAM
from lingam.hsic import hsic_test_gamma
from sklearn.neighbors import KDTree, NearestNeighbors


class _BaseLiNGAM_MMI(_BaseLiNGAM):
    def __init__(self, random_state=None, known_ordering=None):
        super().__init__(random_state)
        self.known_ordering = known_ordering

    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        # Determine topological order
        pi = self._estimate_order(X)
        self._causal_order = pi
        self._estimate_adjacency_matrix(X)
        return self

    def _estimate_order(self, X):
        # using Dijkstra with lazy evaluation of MI
        n_features = X.shape[1]
        start_node = frozenset(range(n_features))
        goal_node = frozenset({})

        OPEN = {start_node}
        distance = {start_node: 0}
        path = {start_node: [start_node]}
        self.mi_estimates = {start_node: []}

        residual_dfs = {self._path2string(path[start_node]): pd.DataFrame(X)}

        self._edges_computed = 0
        while OPEN:
            # Refer to LiNGAM-MMI paper by Suzuki, Algorithm 1
            # 1. Move the node v with the smallest distance from OPEN to CLOSED
            smallest_node = min(OPEN, key=distance.get)
            OPEN -= {smallest_node}

            # 2. Join the successors of v to OPEN
            successors = self._get_successors(smallest_node)
            OPEN.update(successors)

            # 3. If goal_node in OPEN, append goal node to path(v) -- this is the shortest path. Terminate
            if goal_node in OPEN:
                path[smallest_node].append(goal_node)
                self.mi_estimates[smallest_node].append(0)
                return self._path2order(path[smallest_node])

            # 4. Evaluate the mutual information of the successors
            for successor in successors:
                selected_feature_idx = list(smallest_node - successor)[0]
                current_df = residual_dfs[self._path2string(path[smallest_node])]
                residual_df = pd.DataFrame(
                    {
                        c: self._residual(
                            current_df[c], current_df[selected_feature_idx]
                        )
                        for c in current_df.columns
                        if c != selected_feature_idx
                    }
                )
                mi = self._get_mutual_info(
                    current_df, selected_feature_idx, residual_df
                )
                self._edges_computed += 1

                if successor not in distance or (
                    successor in distance
                    and distance[smallest_node] + mi < distance[successor]
                ):
                    distance[successor] = distance[smallest_node] + mi
                    path[successor] = path[smallest_node] + [successor]
                    self.mi_estimates[successor] = self.mi_estimates[smallest_node] + [
                        mi
                    ]
                    residual_dfs[self._path2string(path[successor])] = residual_df

    def _residual(self, dep_var, indep_var):
        """The residual when dep_var is regressed on indep_var"""
        return (
            dep_var - (np.cov(dep_var, indep_var)[0, 1] / np.var(indep_var)) * indep_var
        )

    def _get_successors(self, node: frozenset):
        return {node - {i} for i in node}

    def _get_mutual_info(
        self,
        current_df: pd.DataFrame,
        selected_feature_idx: int,
        residual_df: pd.DataFrame,
    ):
        raise NotImplementedError

    def _path2order(self, path: List[frozenset]):
        path_len = len(path)
        return [list(path[i] - path[i + 1])[0] for i in range(path_len - 1)]

    def _path2string(self, path: List[frozenset]):
        order = self._path2order(path)
        return "_".join([str(i) for i in order])


class kNN_LiNGAM_MMI(_BaseLiNGAM_MMI):
    def __init__(self, k: int = 100, random_state=None, known_ordering=None):
        super().__init__(random_state, known_ordering)
        self.k = k

    def _get_mutual_info(
        self,
        current_df: pd.DataFrame,
        selected_feature_idx: int,
        residual_df: pd.DataFrame,
    ):
        """Compute mutual information between two continuous variables.
        Code copied from sklearn.feature_selection._mutual_info._compute_mi_cc with some modifications

        References
        ----------
        .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
            information". Phys. Rev. E 69, 2004.
        """
        n_neighbors = self.k
        x = np.array(residual_df)
        y = np.array(current_df[selected_feature_idx]).reshape(-1, 1)
        xy = np.hstack((x, y))
        n_samples = x.size

        # Here we rely on NearestNeighbors to select the fastest algorithm.
        nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)

        nn.fit(xy)
        radius = nn.kneighbors()[0]
        radius = np.nextafter(radius[:, -1], 0)

        # KDTree is explicitly fit to allow for the querying of number of
        # neighbors within a specified radius
        kd = KDTree(x, metric="chebyshev")
        nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
        nx = np.array(nx) - 1.0

        kd = KDTree(y, metric="chebyshev")
        ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
        ny = np.array(ny) - 1.0

        mi = (
            digamma(n_samples)
            + digamma(n_neighbors)
            - np.mean(digamma(nx + 1))
            - np.mean(digamma(ny + 1))
        )

        return max(0, mi)


class HSIC_LiNGAM_MMI(_BaseLiNGAM_MMI):
    def __init__(self, use_pval: bool = False, random_state=None, known_ordering=None):
        self.use_pval = use_pval
        super().__init__(random_state, known_ordering)

    def _get_mutual_info(
        self,
        current_df: pd.DataFrame,
        selected_feature_idx: int,
        residual_df: pd.DataFrame,
    ):
        hsic_stat, pval = hsic_test_gamma(
            np.array(current_df[selected_feature_idx]), np.array(residual_df)
        )
        # minimizing 1-pval is equivalent to maximizing pval
        return 1 - pval if self.use_pval else hsic_stat


class PWLR_LiNGAM_MMI(_BaseLiNGAM_MMI):
    def __init__(self, average_mi=True, random_state=None, known_ordering=None):
        self.average_mi = average_mi
        super().__init__(random_state, known_ordering)

    def _get_mutual_info(
        self,
        current_df: pd.DataFrame,
        selected_feature_idx: int,
        residual_df: pd.DataFrame,
    ):
        # we're only using this for the _diff_mutual_info method.
        M = 0
        single_vec = current_df[selected_feature_idx]
        many_vecs = current_df[residual_df.columns]
        xi_std = (single_vec - np.mean(single_vec)) / np.std(single_vec)
        for col_idx in many_vecs.columns:
            xj = many_vecs[col_idx]
            xj_std = (xj - np.mean(xj)) / np.std(xj)
            ri_j = self._residual(xi_std, xj_std)
            rj_i = self._residual(xj_std, xi_std)
            diff_mi = DirectLiNGAM()._diff_mutual_info(xi_std, xj_std, ri_j, rj_i)
            M += np.min([0, diff_mi]) ** 2
        if self.average_mi:
            M /= many_vecs.shape[1]
        return M
