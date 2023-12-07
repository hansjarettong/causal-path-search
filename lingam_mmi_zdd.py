# TODO: refactor this. We can create a base for MutualInfoGraph, similar to what we did for lingam_mmi.py

import itertools
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.special import digamma
from graphillion import GraphSet
import numpy as np
import pandas as pd
from utils import factorial

from lingam.hsic import hsic_test_gamma


class MutualInfoGraph:
    def __init__(self, df, k=100):
        # mutual info dict
        self.mutual_info_dict = dict()
        self.k = k
        self.df = pd.DataFrame(df)

    def _name_node(self, df):
        return "_".join(df.columns.astype(str))

    # def _one2many_mutual_information(self, single_vec, many_vecs):
    #     mi = mutual_info_regression(many_vecs, single_vec, n_neighbors=self.k).sum()
    #     return np.float64(mi)
    def _one2many_mutual_information(self, single_vec, many_vecs):
        """Compute mutual information between two continuous variables.
        Code copied from sklearn.feature_selection._mutual_info._compute_mi_cc

        References
        ----------
        .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
            information". Phys. Rev. E 69, 2004.
        """

        n_neighbors = self.k
        x = np.array(many_vecs)
        y = np.array(single_vec).reshape(-1, 1)
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

    def _residual(self, dep_var, indep_var):
        """The residual when dep_var is regressed on indep_var"""
        return (
            dep_var - np.cov(dep_var, indep_var)[0, 1] / np.var(indep_var) * indep_var
        )

    def _gen_graph_recursuve(self, df):
        this_node = self._name_node(df)
        for selected_col in df.columns:
            resid_df = pd.DataFrame(
                {
                    c: self._residual(df[c], df[selected_col])
                    for c in df.columns
                    if c != selected_col
                }
            )
            self.mutual_info_dict[
                (this_node, self._name_node(resid_df))
            ] = self._one2many_mutual_information(df[selected_col], resid_df)

            if resid_df.shape[1] > 1:
                self._gen_graph_recursuve(resid_df)

    def generate_graph(self, sort=True, save=None):
        self._gen_graph_recursuve(self.df)

        # add in the terminal node
        for c in self.df.columns:
            self.mutual_info_dict[(str(c), "")] = np.float64(0)

        if sort:
            self.mutual_info_dict = dict(
                dict(
                    sorted(
                        self.mutual_info_dict.items(),
                        key=lambda x: (len(x[0][0]), x[0][0], x[0][1]),
                        reverse=True,
                    )
                )
            )

        if save:
            with open(save, "w") as file:
                for key, val in self.mutual_info_dict.items():
                    v1, v2 = key

                    if v1 == "_".join(self.df.columns.astype(str)):
                        v1 = "R"
                    if v1 == "":
                        v1 = "T"
                    if v2 == "_".join(self.df.columns.astype(str)):
                        v2 = "R"
                    if v2 == "":
                        v2 = "T"

                    file.write(f"{v1} {v2} {val}\n")

        return self.mutual_info_dict


class HSICMutualInfoGraph(MutualInfoGraph):
    def _one2many_mutual_information(self, single_vec, many_vecs):
        single_vec = np.array(single_vec)
        many_vecs = np.array(many_vecs)
        hsic_stat, pval = hsic_test_gamma(single_vec, many_vecs)
        return hsic_stat


class PWLINGMutualInfoGraph(MutualInfoGraph):
    def _diff_mutual_info(self, xi_std, xj_std, ri_j, rj_i):
        """Calculate the difference of the mutual informations."""
        return (self._entropy(xj_std) + self._entropy(ri_j / np.std(ri_j))) - (
            self._entropy(xi_std) + self._entropy(rj_i / np.std(rj_i))
        )

    def _entropy(self, u):
        """Calculate entropy using the maximum entropy approximations."""
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (
            (1 + np.log(2 * np.pi)) / 2
            - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2
            - k2 * (np.mean(u * np.exp((-(u**2)) / 2))) ** 2
        )

    def _one2many_mutual_information(self, single_vec, many_vecs):
        M = 0
        xi_std = (single_vec - np.mean(single_vec)) / np.std(single_vec)
        for col_idx in many_vecs.columns:
            xj = many_vecs[col_idx]
            xj_std = (xj - np.mean(xj)) / np.std(xj)
            ri_j = self._residual(xi_std, xj_std)
            rj_i = self._residual(xj_std, xi_std)
            diff_mi = self._diff_mutual_info(xi_std, xj_std, ri_j, rj_i)
            M += np.min([0, diff_mi]) ** 2
        return M / many_vecs.shape[1]

    def _gen_graph_recursuve(self, df):
        this_node = self._name_node(df)
        for selected_col in df.columns:
            resid_df = pd.DataFrame(
                {
                    c: self._residual(df[c], df[selected_col])
                    for c in df.columns
                    if c != selected_col
                }
            )
            self.mutual_info_dict[
                (this_node, self._name_node(resid_df))
            ] = self._one2many_mutual_information(
                df[selected_col], df[resid_df.columns]
            )

            if resid_df.shape[1] > 1:
                self._gen_graph_recursuve(resid_df)


class ZDD:
    def __init__(self, mi_dict, known_ordering=None, traversal="as-is"):
        self.nodes_to_exclude = set()

        if known_ordering:
            assert (
                len(known_ordering) >= 2
            ), "known_ordering should have at least 2 items."

        self.mi_dict = mi_dict

        universe = [k + tuple([v]) for k, v in mi_dict.items()]

        GraphSet.set_universe(universe, traversal=traversal)
        GraphSet.show_messages()

        root_node = self._get_root_node()
        self.num_features = len(self._nodestr2list(root_node))

        # setting the graph_size first will reduce memory and runtime costs but may kill the kernel at times
        self.paths = (
            GraphSet({}).graph_size(self.num_features).paths(self._get_root_node(), "")
        )

        if known_ordering:
            # enumerate all pairs and do the exclusion pair-wise
            for i, j in itertools.combinations(known_ordering, 2):
                self._exclude_nodes_from_known_pair_order(i, j)

        for node in self.nodes_to_exclude:
            self.paths = self.paths.excluding(node)

    def _get_root_node(self):
        # get the longest node string
        return max(self.mi_dict.keys(), key=lambda x: len(x[0]))[0]

    def _nodestr2list(self, nodestr):
        if nodestr:
            return [int(i) for i in nodestr.split("_")]
        return []

    def _list2nodestr(self, list_):
        return "_".join([str(i) for i in list_])

    def _path2ordering(self, path):
        path = sorted(path, reverse=True, key=lambda x: len(x[0]))
        order = []
        for first, second in path:
            first = set(self._nodestr2list(first))
            second = set(self._nodestr2list(second))
            # this should be a single-element set anyway
            diff = first - second
            assert len(diff) == 1, "Invalid Path"
            order.append(min(diff))
        return order

    def _get_path_mi(self, path):
        return sum([self.mi_dict[k] for k in path])

    def _exclude_nodes_from_known_pair_order(self, earlier_var, later_var):
        # don't pass through nodes that violate the known pair order
        for r in range(self.num_features):
            self.nodes_to_exclude = self.nodes_to_exclude.union(
                {
                    self._list2nodestr(i)
                    for i in itertools.combinations(range(self.num_features), r)
                    if earlier_var in i and later_var not in i
                }
            )

    def get_shortest_path_ordering(self):
        paths_miniter = iter(self.paths.min_iter())
        shortest_path = next(paths_miniter)
        return self._path2ordering(shortest_path)

    def get_mi_per_ordering(self, sample_size=-1) -> dict:
        assert sample_size <= factorial(
            self.num_features
        ), "Sample size is larger than the number of paths."
        paths = (
            itertools.islice(self.paths.rand_iter(), sample_size)
            if sample_size > 0
            else self.paths
        )
        ord_mi_dict = dict()
        for path in paths:
            causal_ordering = self._path2ordering(path)
            total_mi = self._get_path_mi(path)
            ord_mi_dict[tuple(causal_ordering)] = total_mi

        return ord_mi_dict
