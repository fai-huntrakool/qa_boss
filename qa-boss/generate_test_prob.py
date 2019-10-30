from pyqubo import Array, Constraint
import numpy as np
import solver


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


def construct_bqm(x, lamb, numerator, divisor, with_con=0):
    feed_obj_fun = (np.sum(np.dot(numerator, x)) - np.sum(np.dot(lamb, np.dot(divisor, x))))
    if with_con == 1:
        constraint = Constraint(np.sum((x - 1) * (x - 1)), label='test')
        feed_obj_fun = feed_obj_fun + constraint
    model = feed_obj_fun.compile()
    bqm = model.to_dimod_bqm()
    return bqm


import dinkelbach as dkb

if __name__ == '__main__':
    size = 3
    x, numerator, divisor = generate_test_case(size, 1)
    lamb, _, _, _, _, previous_solution, _, _ = dkb.initialize_lambda(1, size, numerator, divisor)
    bqm = construct_bqm(x, lamb, numerator, divisor, 1)
    ans, check = solver.sa_solver(bqm, previous_solution)
