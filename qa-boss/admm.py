import generate_test_prob as gtp
import dinkelbach as dkb
import solver
import numpy as np


def find_min_xi(x, numerator, divisor, lamb, previous_solution):
    x_i = []
    for num, div, l in zip(numerator, divisor, lamb):
        solution, _, _ = dkb.dinkelbach_for_one_ratio(x, [l], [num], [div], previous_solution)
        x_i.append(solution)
        print('Exact Solver : ')
        print(solver.exact_solver([num], [div], 3, 1))
    return x_i


def update_z(x_i, p, y_i):
    mat_x = np.array(x_i)
    avg_x_i = mat_x.mean(0)

    mat_y = np.array(y_i)
    avg_y_i = mat_y.mean(0)
    return avg_x_i + 1 / p * avg_y_i


def update_y(x_i, y_i, z, p):
    return y_i + p * (x_i - z)


if __name__ == '__main__':
    size = 3
    num_terms = 3
    x, numerator, divisor = gtp.generate_test_case(3, num_terms)
    lamb, _, _, _, _, previous_solution, _, _ = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    #initialize
    p = 0
    y_i = np.array([0]*size)

    while 1 < 0: #stop condition
        x_i = find_min_xi(x, numerator, divisor, lamb, previous_solution)
        z = update_z(x_i, p, y_i)
        y_i = update_y(x_i, y_i, z, p)
