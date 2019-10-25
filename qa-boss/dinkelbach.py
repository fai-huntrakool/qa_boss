import numpy as np
import generate_test_prob as gtp
import solver


def objective_value(x, numerator, divisor, lamb):
    # obj_1 = num/den
    # obj_2 = num - lamb*den

    num = np.dot(numerator, x)
    den = np.dot(divisor, x)
    den_with_lamb = lamb * den

    sub_obj_1 = [num[i] / den[i] if den[i] != 0 else float('Inf') for i in range(len(num))]
    obj_1 = np.sum(sub_obj_1)

    sub_obj_2 = [num[i] - den_with_lamb[i] for i in range(len(num))]
    obj_2 = np.sum(sub_obj_2)
    return obj_1, sub_obj_1, obj_2, sub_obj_2


def dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution, i=1):
    bqm = gtp.construct_bqm(x, lamb, numerator, divisor)
    current_solution = solver.sa_solver(bqm, previous_solution)
    obj_1, sub_obj_1, obj_2, _ = objective_value(current_solution, numerator, divisor, lamb)
    previous_solution = current_solution

    print('iteration: {}'.format(i))
    print(current_solution)
    print('min P(x)-a*Q(X) = {}.'.format(obj_2))
    print('fix a={} : min P(x)/Q(X) = {}.'.format(lamb, obj_1))
    print('--------------------------------------------')

    if obj_2 != 0:
        lamb = np.array(sub_obj_1)
        dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution, i+1)
    else:
        return x, obj_1, obj_2



