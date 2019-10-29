import dinkelbach as dkb
import generate_test_prob as gtp
import solver


def one_ratio(size, is_enable=0, num_terms=1):
    x, numerator, divisor = gtp.generate_test_case(size, num_terms)
    lamb, _, _, _, _, previous_solution = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    dkb.dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution)
    if is_enable == 1:
        print('Exact Solver : ')
        print(solver.exact_solver(numerator, divisor, size, num_terms))


def multiple_ratio(size, num_terms, limit_iteration=150):
    x, numerator, divisor = gtp.generate_test_case(size, num_terms)
    lamb, u, v, uk, vk, previous_solution = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    dkb.dinkelbach_for_multiple_ratios(x, lamb, u, v, uk, vk, numerator, divisor, previous_solution, limit_iteration, 1)
    print('Exact Solver : ')
    print(solver.exact_solver(numerator, divisor, size, num_terms))


if __name__ == '__main__':
    one_ratio(15, 1)
    #multiple_ratio(3, 3, 20)
