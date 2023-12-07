"""
LiNGAM Data Generation Script

This Python script implements the data generating process described in the paper:
"DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model"
by Shohei Shimizu, et al. with some modifications to incorporate unmeasured confounders.

Please make sure to cite the original paper if you use or reference this script in your work.
"""

import numpy as np
from scipy.stats import t, laplace, uniform, expon, norm


def generate_adjacency_matrix(num_features, sparseness=0.1):
    p = num_features
    # random array in the sampled from [−1.5,−0.5] ∪ [0.5,1.5]
    arr = np.random.uniform(0.5, 1.5, (p, p)) * (np.random.choice([0, 2], (p, p)) - 1)
    # mask to make it lower triangular
    mask = np.random.binomial(1, 1 - sparseness, (p, p)) * np.tri(p, p, k=-1, dtype=int)
    return mask * arr


def generate_data_from_matrix(
    B: np.array,
    sample_size=1000,
    num_confounders=1,
    confoundedness=0.5,
    confounder_variance=None,
):
    assert np.all(
        np.triu(B) == 0
    ), "The adjacency matrix should be strictly lower triangular."

    p = B.shape[0]  # num_features
    n = sample_size

    confounder_matrix = np.random.binomial(1, confoundedness, size=(p, num_confounders))
    confounders = np.array(
        [
            NonGaussianDistributions().generate_non_gaussian_noise(
                n, var=confounder_variance
            )
            for _ in range(num_confounders)
        ]
    ).T

    X = np.zeros((n, p))
    for i, row in enumerate(B):
        col_i = (
            (X @ row).reshape(-1, 1)
            + NonGaussianDistributions().generate_non_gaussian_noise(n).reshape(-1, 1)
            + (confounders @ confounder_matrix[i]).reshape(-1, 1)
        )
        X[:, i] = col_i.flatten()

    return X


def generate_data(
    num_features,
    sparseness=0.1,
    sample_size=1000,
    num_confounders=1,
    confounder_strength=1,
    confoundedness=0,
):
    B = generate_adjacency_matrix(num_features, sparseness)
    p = B.shape[0]  # num_features
    n = sample_size

    confounders = [
        confounder_strength * NonGaussianDistributions().generate_non_gaussian_noise(n)
        for _ in range(num_confounders)
    ]

    features_to_confound = []
    while len(features_to_confound) < len(confounders):
        # must at least be 2
        features = set(np.random.choice(range(num_features), 2, replace=False))
        # add a random number of random features
        other_features = np.where(np.random.binomial(1, confoundedness, num_features))[
            0
        ]
        features = features.union(other_features)
        if features not in features_to_confound:
            features_to_confound.append(features)

    X = np.zeros((n, p))
    for i, row in enumerate(B):
        col_i = (X @ row).reshape(
            -1, 1
        ) + NonGaussianDistributions().generate_non_gaussian_noise(n).reshape(-1, 1)
        for conf_idx in range(num_confounders):
            if i in features_to_confound[conf_idx]:
                col_i += confounders[conf_idx].reshape(-1, 1)
        X[:, i] = col_i.flatten()

    return X


class NonGaussianDistributions:
    def __init__(self):
        self.distributions = [
            self.student_3_dof,
            self.double_exponential,
            self.uniform,
            self.student_5_dof,
            self.exponential,
            self.mixture_of_double_exponentials,
            self.symmetric_mixture_of_two_Gaussians_multimodal,
            self.symmetric_mixture_of_two_Gaussians_transitional,
            self.symmetric_mixture_of_two_Gaussians_unimodal,
            self.nonsymmetric_mixture_of_two_Gaussians_multimodal,
            self.nonsymmetric_mixture_of_two_Gaussians_transitional,
            self.nonsymmetric_mixture_of_two_Gaussians_unimodal,
            self.symmetric_mixture_of_four_Gaussians_multimodal,
            self.symmetric_mixture_of_four_Gaussians_transitional,
            self.symmetric_mixture_of_four_Gaussians_unimodal,
            self.nonsymmetric_mixture_of_four_Gaussians_multimodal,
            self.nonsymmetric_mixture_of_four_Gaussians_transitional,
            self.nonsymmetric_mixture_of_four_Gaussians_unimodal,
        ]

    def generate_non_gaussian_noise(self, n, var=None):
        if var is None:
            # sample noise variance from [1,3] as was done by Shimizu et al. 2011 and Silva et al. 2006
            var = np.random.uniform(1, 3)

        noise = np.random.choice(self.distributions)(n)
        return noise / noise.std() * np.sqrt(var)

    def student_3_dof(self, n):
        return t(3).rvs(n)

    def double_exponential(self, n):
        return laplace().rvs(n)

    def uniform(self, n):
        return uniform(-1, 1).rvs(n)

    def student_5_dof(self, n):
        return t(5).rvs(n)

    def exponential(self, n):
        return expon().rvs(n)

    def mixture_of_double_exponentials(self, n):
        laplace1 = laplace(loc=-3).rvs(n)
        laplace2 = laplace(loc=3).rvs(n)
        p = np.random.binomial(1, 0.5, n)
        return laplace1 * p + laplace2 * (1 - p)

    def symmetric_mixture_of_two_Gaussians_multimodal(self, n):
        norm1 = np.random.normal(-3, 1, n)
        norm2 = np.random.normal(3, 1, n)
        p = np.random.binomial(1, 0.5, n)
        return norm1 * p + norm2 * (1 - p)

    def symmetric_mixture_of_two_Gaussians_transitional(self, n):
        norm1 = np.random.normal(-1.5, 1, n)
        norm2 = np.random.normal(1.5, 1, n)
        p = np.random.binomial(1, 0.5, n)
        return norm1 * p + norm2 * (1 - p)

    def symmetric_mixture_of_two_Gaussians_unimodal(self, n):
        norm1 = np.random.normal(-1, 1, n)
        norm2 = np.random.normal(1, 1, n)
        p = np.random.binomial(1, 0.5, n)
        return norm1 * p + norm2 * (1 - p)

    def nonsymmetric_mixture_of_two_Gaussians_multimodal(self, n):
        norm1 = np.random.normal(-3, 1, n)
        norm2 = np.random.normal(3, 1, n)
        p = np.random.binomial(1, 0.25, n)
        return norm1 * p + norm2 * (1 - p)

    def nonsymmetric_mixture_of_two_Gaussians_transitional(self, n):
        norm1 = np.random.normal(-1.5, 1, n)
        norm2 = np.random.normal(1.5, 1, n)
        p = np.random.binomial(1, 0.25, n)
        return norm1 * p + norm2 * (1 - p)

    def nonsymmetric_mixture_of_two_Gaussians_unimodal(self, n):
        norm1 = np.random.normal(-1, 1, n)
        norm2 = np.random.normal(1, 1, n)
        p = np.random.binomial(1, 0.25, n)
        return norm1 * p + norm2 * (1 - p)

    def symmetric_mixture_of_four_Gaussians_multimodal(self, n):
        norms = np.vstack([np.random.normal(mu, 1, n) for mu in [-6, -2, 2, 6]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.15, 0.35, 0.35, 0.15])]
        return (norms.T * mask).sum(axis=1)

    def symmetric_mixture_of_four_Gaussians_transitional(self, n):
        norms = np.vstack([np.random.normal(mu, 1, n) for mu in [-4.5, -1, 1, 4.5]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.15, 0.35, 0.35, 0.15])]
        return (norms.T * mask).sum(axis=1)

    def symmetric_mixture_of_four_Gaussians_unimodal(self, n):
        norms = np.vstack([np.random.normal(mu, 1, n) for mu in [-3.8, -1, 1, 3.8]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.15, 0.35, 0.35, 0.15])]
        return (norms.T * mask).sum(axis=1)

    def nonsymmetric_mixture_of_four_Gaussians_multimodal(self, n):
        norms = np.vstack([np.random.normal(mu, 1, n) for mu in [-6, -2, 1.25, 6]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.20, 0.40, 0.20, 0.20])]
        return (norms.T * mask).sum(axis=1)

    def nonsymmetric_mixture_of_four_Gaussians_transitional(self, n):
        norms = np.vstack([np.random.normal(mu, 1, n) for mu in [-4.5, -1.2, 1, 4.5]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.2, 0.3, 0.4, 0.1])]
        return (norms.T * mask).sum(axis=1)

    def nonsymmetric_mixture_of_four_Gaussians_unimodal(self, n):
        norms = np.vstack([np.random.normal(mu, 1, n) for mu in [-3, -1, 1, 3]])
        mask = np.eye(4)[np.random.choice(4, n, p=[0.15, 0.15, 0.35, 0.35])]
        return (norms.T * mask).sum(axis=1)
