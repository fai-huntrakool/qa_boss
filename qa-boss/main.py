import dinkelbach as dkb
import generate_test_prob as gtp
import solver
from scipy.optimize import line_search
import generate_prob as gp


def one_ratio(size, is_enable=0, num_terms=1):
    x, numerator, divisor = gtp.generate_test_case(size, num_terms)
    lamb, _, _, _, _, previous_solution, _, _ = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    dkb.dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution)
    if is_enable == 1:
        print('Exact Solver : ')
        print(solver.exact_solver(numerator, divisor, size, num_terms))


def multiple_ratio(size, num_terms, limit_iteration=150):
    x, numerator, divisor = gtp.generate_test_case(size, num_terms)
    print('Exact Solver : ')
    print(solver.exact_solver(numerator, divisor, size, num_terms))
    print('--------------------------------------------')
    lamb, u, v, uk, vk, previous_solution, obj_1, obj_2 = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    if obj_2 == 0:
        dkb.print_iteration_value(0, previous_solution, obj_1, obj_2, lamb)
    else:
        dkb.dinkelbach_for_multiple_ratios(x, lamb, u, v, uk, vk, numerator, divisor, previous_solution,
                                           limit_iteration,0, 1)
    print('Exact Solver : ')
    print(solver.exact_solver(numerator, divisor, size, num_terms))


def mock_problem(size, num_covs, num_bins, limit_iteration = 100):

    num_terms = (num_bins-1) * num_covs
    df = gp.create_random_dataset(size, num_covs)
    b_mat = gp.create_bins_array(df, num_bins, num_covs)
    x, numerator, divisor = gp.problem_formulation(size, b_mat, num_bins)
    print(numerator)
    print(divisor)
    lamb, u, v, uk, vk, previous_solution, obj_1, obj_2 = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    if obj_2 == 0:
        dkb.print_iteration_value(0, previous_solution, obj_1, obj_2, lamb)
    else:
        dkb.dinkelbach_for_multiple_ratios(x, lamb, u, v, uk, vk, numerator, divisor, previous_solution,
                                           limit_iteration,is_spin=1,i=1)
    print('Exact Solver : ')
    print(solver.exact_solver(numerator, divisor, size, num_terms, is_spin = 1))


if __name__ == '__main__':
    one_ratio(3, 1)
    #multiple_ratio(10, 1, 150)
    #mock_problem(size = 10, num_covs=2, num_bins=3)