import numpy as np
from pyqubo import Array


# size = no. of decision variables
# num_terms = no. of fractions
# num_bins = no. of bins

def create_random_dataset(size, num_covs):
    return np.random.random((size, num_covs)) * 10


def create_bins_array(initial_mat, num_bins, n_cov):
    b_mat = np.zeros((len(initial_mat), len(initial_mat[0])))
    bins = np.linspace(0, 10, num_bins)
    for b in range(n_cov):
        data = initial_mat[:, b]
        digitized = np.digitize(data, bins)
        b_mat[:, b] = digitized
    return b_mat


def problem_formulation(size, b_mat, num_bins):
    S = np.append(Array.create('s', shape=size, vartype='SPIN'), [1])
    numerator, divisor = [], []
    for col in range(len(b_mat[0])):
        for b in range(1, num_bins):
            nume, div = [], []
            for val in b_mat[:, col]:
                if val == b:
                    nume.append(1)
                    div.append(-1)
                else:
                    nume.append(0)
                    div.append(0)
            nume.append(0)
            div.append((sum(div) * -1) + 0.1)
            numerator.append(nume)
            divisor.append(div)
    return S, numerator, divisor
