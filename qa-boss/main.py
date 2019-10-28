import dinkelbach as dkb
import generate_test_prob as gtp
import solver
import numpy as np


def main():
    size = 3
    num_terms = 1

    x, numerator, divisor = gtp.generate_test_case(size, num_terms)

    lamb, _, _, _, _, previous_solution = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    dkb.dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution)
    print('Exact Solver : ')
    print(solver.exact_solver(numerator, divisor, size, num_terms))


def multiple_ratio():
    size = 3
    num_terms = 3

    x, numerator, divisor = gtp.generate_test_case(size, num_terms)
    lamb, u, v, uk, vk, previous_solution = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    dkb.dinkelbach_for_multiple_ratios(x, lamb, u, v, uk, vk, numerator, divisor, previous_solution, 1)
    print('Exact Solver : ')
    print(solver.exact_solver(numerator, divisor, size, num_terms))


if __name__ == '__main__':
    multiple_ratio()