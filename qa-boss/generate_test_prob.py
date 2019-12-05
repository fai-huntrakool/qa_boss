from pyqubo import Array, Constraint
import numpy as np
import solver


# generate test problem

def generate_test_case(size, num_terms, var_type="BINARY"):
    x = np.append(Array.create('x', size, var_type), [1])
    numerator, divisor = [], []
    for i in range(num_terms):
       # np.random.seed(700+50*i)
        numerator.append(np.random.randint(1, 10, size + 1))
      #  np.random.seed(500+10*i)
        divisor.append(np.random.randint(1, 10, size + 1))
    print(numerator)
    print(divisor)
    return x, numerator, divisor


def initialize_solution(size):
    return np.random.randint(0, 2, size=size)


def construct_bqm(x, lamb, numerator, divisor):
    feed_obj_fun = (np.sum(np.dot(numerator, x)) - np.sum(np.dot(lamb, np.dot(divisor, x))))
    model = feed_obj_fun.compile()
    bqm = model.to_dimod_bqm()
    return bqm


import dinkelbach as dkb

if __name__ == '__main__':
    size = 3
    num_terms = 1
    var_type = 'BINARY'
    x = np.append(Array.create('x', size, var_type), [1])
    y = Array.create('y', size, var_type)
    numerator, divisor = [], []
    for i in range(num_terms):
        numerator.append(np.random.randint(1, 10, size + 1, 700))
        divisor.append(np.random.randint(1, 10, size + 1, 700))
    lamb, _, _, _, _, previous_solution, _, _ = dkb.initialize_lambda(1, size, numerator, divisor)
    feed_obj_fun = (np.sum(np.dot(numerator, x)) - np.sum(np.dot(lamb, np.dot(divisor, x))))
    model = feed_obj_fun.compile()
    bqm_x = model.to_dimod_bqm()

    print(numerator[0][:-1])
    feed_obj_fun = (np.sum(np.dot(numerator[0][:-1], y)) - np.sum(np.dot(lamb, np.dot(divisor[0][:-1], y))))
    model = feed_obj_fun.compile()
    bqm_y = model.to_dimod_bqm()

    print(bqm_x)
    print(bqm_y)
