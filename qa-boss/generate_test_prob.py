from pyqubo import Array
import numpy as np
import solver
import dinkelbach


# generate test problem

def generate_test_case(size, num_terms, var_type="BINARY"):
    x = Array.create('x', size, var_type)
    numerator, divisor = [], []
    for i in range(num_terms):
        numerator.append(np.random.randint(1, 10, size))
        divisor.append(np.random.randint(1, 10, size))
    return x, numerator, divisor


def initialize_solution(size):
    return np.random.randint(0, 2, size=size)


def construct_bqm(x, lamb, numerator, divisor):
    feed_obj_fun = np.sum(np.dot(numerator, x)) - np.sum(np.dot(lamb, np.dot(divisor, x)))
    model = feed_obj_fun.compile()
    bqm = model.to_dimod_bqm()
    return bqm

