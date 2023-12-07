import numpy as np
import pandas as pd
import pickle
from scipy.special import comb


def ordering_score(causal_order):
    # i.e. number of incorrect pairs
    count = 0
    for i in range(len(causal_order)):
        for j in range(i + 1, len(causal_order)):
            if causal_order[i] > causal_order[j]:
                count += 1
    return count / comb(len(causal_order), 2)


def get_kth_std_moment(X, k=3):
    X = X - X.mean()
    mu_k = (X**k).mean()
    sigma_k = (X**2).mean() ** (k / 2)
    return mu_k / sigma_k


def sim_pickle2frame(results_pickle_loc, max_std_moment=20):
    with open(results_pickle_loc, "rb") as f:
        results = pickle.load(f)
    for result in results:
        mi = result["mi_all_orderings"].values()
        mi = np.array(list(mi))
        for k in range(3, max_std_moment + 1):
            result[f"std_moment_{str(k).zfill(2)}"] = get_kth_std_moment(mi, k)
    return pd.DataFrame(results).assign(
        shortest_path_ordering=lambda x: x.shortest_path_ordering.map(tuple),
        all_paths=lambda x: x.mi_all_orderings.map(lambda y: list(y.values())),
        ordering_error_score=lambda x: x.shortest_path_ordering.map(ordering_score),
        ordering_is_correct=lambda x: (
            x.shortest_path_ordering == x.num_features.map(lambda y: tuple(range(y)))
        ).astype(int),
        has_confounders=lambda x: (x.confounder_strength != 0).astype(int),
    )


def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    elif n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def frobenius_norm(matrix1, matrix2):
    # Ensure both matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError(
            "Matrices must have the same shape for Frobenius norm calculation"
        )

    # Compute the element-wise squared difference between the matrices
    squared_diff = (matrix1 - matrix2) ** 2

    # Sum all the squared differences and take the square root
    norm = np.sqrt(np.sum(squared_diff))

    return norm
