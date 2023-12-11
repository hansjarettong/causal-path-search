from itertools import combinations
import itertools
from utils import factorial
import pandas as pd
import numpy as np
from graphillion import GraphSet

# reuse the _get_mutual_info implementations from lingam_mmi.py
from lingam_mmi import kNN_LiNGAM_MMI, HSIC_LiNGAM_MMI, PWLR_LiNGAM_MMI


class _BaseLiNGAM_ZDD:
    def __init__(self, known_ordering: list = None, traversal="as-is"):
        if not bool(known_ordering):
            self.known_ordering = list()
        elif isinstance(known_ordering[0], int):
            self.known_ordering = list(combinations(known_ordering, 2))
        else:
            self.known_ordering = known_ordering

        self.traversal = traversal
        self.nodes_to_exclude = set()
        self.mi_dict = dict()

    def fit(self, df: pd.DataFrame):
        self._get_mi_dict(df)

        universe = [k + tuple([v]) for k, v in self.mi_dict.items()]

        GraphSet.set_universe(universe, traversal=self.traversal)

        root_node = self._get_root_node()
        self.num_features = len(self._nodestr2list(root_node))

        # setting the graph_size first will reduce memory and runtime costs but may kill the kernel at times
        self.paths = (
            GraphSet({}).graph_size(self.num_features).paths(self._get_root_node(), "")
        )

        if self.known_ordering:
            # enumerate all pairs and do the exclusion pair-wise
            for i, j in self.known_ordering:
                self._exclude_nodes_from_known_pair_order(i, j)

        for node in self.nodes_to_exclude:
            self.paths = self.paths.excluding(node)

        return self

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
                    for i in combinations(range(self.num_features), r)
                    if earlier_var in i and later_var not in i
                }
            )

    def _name_node(self, df: pd.DataFrame):
        # we have to do this because the ZDD backend explects a string as node names
        return "_".join(df.columns.astype(str))

    def _get_mutual_info(
        self,
        current_df: pd.DataFrame,
        selected_feature_idx: int,
        residual_df: pd.DataFrame,
    ):
        raise NotImplementedError

    def _residual(self, dep_var, indep_var):
        """The residual when dep_var is regressed on indep_var"""
        return (
            dep_var - (np.cov(dep_var, indep_var)[0, 1] / np.var(indep_var)) * indep_var
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
            self.mi_dict[
                (this_node, self._name_node(resid_df))
            ] = self._get_mutual_info(df, selected_col, resid_df)

            if resid_df.shape[1] > 1:
                self._gen_graph_recursuve(resid_df)

    def _get_mi_dict(self, df: pd.DataFrame, sort=True, save=None) -> dict:
        self._gen_graph_recursuve(df)

        # add in the terminal node
        for c in df.columns:
            self.mi_dict[(str(c), "")] = np.float64(0)

        if sort:
            self.mi_dict = dict(
                dict(
                    sorted(
                        self.mi_dict.items(),
                        key=lambda x: (len(x[0][0]), x[0][0], x[0][1]),
                        reverse=True,
                    )
                )
            )

        if save:
            with open(save, "w") as file:
                for key, val in self.mi_dict.items():
                    v1, v2 = key

                    if v1 == "_".join(df.columns.astype(str)):
                        v1 = "R"
                    if v1 == "":
                        v1 = "T"
                    if v2 == "_".join(df.columns.astype(str)):
                        v2 = "R"
                    if v2 == "":
                        v2 = "T"

                    file.write(f"{v1} {v2} {val}\n")

        return self.mi_dict


class kNN_LiNGAM_ZDD(_BaseLiNGAM_ZDD):
    def __init__(self, k: int = 100, known_ordering: list = None):
        self.k = k
        super().__init__(known_ordering)

    def _get_mutual_info(
        self,
        current_df: pd.DataFrame,
        selected_feature_idx: int,
        residual_df: pd.DataFrame,
    ):
        return kNN_LiNGAM_MMI(self.k)._get_mutual_info(
            current_df, selected_feature_idx, residual_df
        )


class HSIC_LiNGAM_ZDD(_BaseLiNGAM_ZDD):
    def __init__(self, use_pval: bool = False, known_ordering: list = None):
        self.use_pval = use_pval
        super().__init__(known_ordering)

    def _get_mutual_info(
        self,
        current_df: pd.DataFrame,
        selected_feature_idx: int,
        residual_df: pd.DataFrame,
    ):
        return HSIC_LiNGAM_MMI(self.use_pval)._get_mutual_info(
            current_df, selected_feature_idx, residual_df
        )


class PWLR_LiNGAM_ZDD(_BaseLiNGAM_ZDD):
    def __init__(self, average_mi: bool = True, known_ordering: list = None):
        self.average_mi = average_mi
        super().__init__(known_ordering)

    def _get_mutual_info(
        self,
        current_df: pd.DataFrame,
        selected_feature_idx: int,
        residual_df: pd.DataFrame,
    ):
        return PWLR_LiNGAM_MMI(self.average_mi)._get_mutual_info(
            current_df, selected_feature_idx, residual_df
        )
