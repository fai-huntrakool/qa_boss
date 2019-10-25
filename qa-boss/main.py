import dinkelbach
import generate_test_prob as gtp
import solver


def main():
    size = 15
    num_terms = 1
    x, numerator, divisor = gtp.generate_test_case(size, num_terms)
    lamb = gtp.initialize_lambda(num_terms)
    previous_solution = gtp.initialize_solution(size)
    dinkelbach.dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution)
    print(solver.exact_solver(numerator, divisor, size, num_terms))


if __name__ == '__main__':
    main()
