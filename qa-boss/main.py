import dinkelbach as dkb
import generate_test_prob as gtp
import solver
from scipy.optimize import line_search

def one_ratio(size, is_enable=0, num_terms=1):
    x, numerator, divisor = gtp.generate_test_case(size, num_terms)
    lamb, _, _, _, _, previous_solution, _, _ = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    dkb.dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution)
    if is_enable == 1:
        print('Exact Solver : ')
        print(solver.exact_solver(numerator, divisor, size, num_terms))


def multiple_ratio(size, num_terms, limit_iteration=150):
    x, numerator, divisor = gtp.generate_test_case(size, num_terms)
    lamb, u, v, uk, vk, previous_solution, obj_1, obj_2 = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    if obj_2 == 0:
        dkb.print_iteration_value(0, previous_solution, obj_1, obj_2, lamb)
    else:
        dkb.dinkelbach_for_multiple_ratios(x, lamb, u, v, uk, vk, numerator, divisor, previous_solution,
                                           limit_iteration, 1)
    print('Exact Solver : ')
    print(solver.exact_solver(numerator, divisor, size, num_terms))


import numpy as np

if __name__ == '__main__':
    # one_ratio(3, 1)
    #
    multiple_ratio(3, 3, 100)
    # size = 3
    # num_terms = 3
    #
    # x, numerator, divisor = gtp.generate_test_case(size, num_terms)
    # lamb, u, v, uk, vk, previous_solution, obj_1, obj_2 = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    #
    # bqm = gtp.construct_bqm(x, lamb, numerator, divisor)
    # current_solution = solver.sa_solver(bqm, previous_solution)
    # obj_1, sub_obj_1, obj_2, sub_obj_2, num, den = dkb.objective_value(current_solution, numerator, divisor, lamb)
    # hess, h_func = dkb.hessian_matrix(den)
    #
    # lamb = dkb.update_lambda_tr(lamb, num, den)
    #print(lamb)
    # f_value, func = dkb.f(lamb, num, den)
    # delta, d_func = dkb.delta_vector(lamb, num, den)
    # pt = dkb.descent_condition(delta)
    # alpha, _, _, _, _, _ = line_search(func, d_func, lamb, pt)
    # print("alpha", lamb+alpha*pt)
    # print(dkb.update_lambda_ls(lamb, num, den))
