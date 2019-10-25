import dinkelbach as dkb
import generate_test_prob as gtp
import solver


def main():
    size = 15
    num_terms = 1

    x, numerator, divisor = gtp.generate_test_case(size, num_terms)

    lamb, _, _, _, _, _ = dkb.initialize_lambda(num_terms, size, numerator, divisor)
    previous_solution = gtp.initialize_solution(size)

    dkb.dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution)
    print('Exact Solver : ' )
    print(solver.exact_solver(numerator, divisor, size, num_terms))

def multiple_ratio():
    size = 3
    num_terms = 3

    x, numerator, divisor = gtp.generate_test_case(size, num_terms)

    lamb, v, u, vk, uk, previous_solution = dkb.initialize_lambda(num_terms, size, numerator, divisor)

    dkb.dinkelbach_for_multiple_ratios(x, lamb, u, v, numerator, divisor, previous_solution)
    print('Exact Solver : ')
    print(solver.exact_solver(numerator, divisor, size, num_terms))


if __name__ == '__main__':
    multiple_ratio()
