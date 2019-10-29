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


def print_iteration_value(i, current_solution, obj_1, obj_2, lamb):
    print('iteration: {}'.format(i))
    print(current_solution)
    print('min P(x)-a*Q(X) = {}.'.format(obj_2))
    print('fix lambda={} : min P(x)/Q(X) = {}.'.format(lamb, obj_1))
    print('--------------------------------------------')


def dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution, i=1):
    bqm = gtp.construct_bqm(x, lamb, numerator, divisor)
    current_solution = solver.sa_solver(bqm, previous_solution)
    obj_1, sub_obj_1, obj_2, _ = objective_value(current_solution, numerator, divisor, lamb)
    previous_solution = current_solution
    print_iteration_value(i, current_solution, obj_1, obj_2, lamb)

    if obj_2 != 0:
        lamb = np.array(sub_obj_1)
        dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution, i + 1)
    else:
        return x, obj_1, obj_2


def dinkelbach_for_multiple_ratios(x, lamb, u, v, uk, vk, numerator, divisor, previous_solution, limit_iteration, i=1):
    bqm = gtp.construct_bqm(x, lamb, numerator, divisor)
    current_solution = solver.sa_solver(bqm, previous_solution)
    obj_1, sub_obj_1, obj_2, sub_obj_2 = objective_value(current_solution, numerator, divisor, lamb)
    previous_solution = current_solution
    print_iteration_value(i, current_solution, obj_1, obj_2, lamb)

    if (abs(obj_2) <= 0.0001) | (i > limit_iteration):
        return x, obj_1, obj_2
    elif obj_2 != 0:
        lamb, u, v = update_lambda(lamb, u, v, uk, vk, obj_2, sub_obj_1, sub_obj_2)
        dinkelbach_for_multiple_ratios(x, lamb, u, v, uk, vk, numerator, divisor, previous_solution, limit_iteration,
                                       i+1)


def check_if_neg(l):
    for i in l:
        if i < 0:
            return False
    return True


def initialize_lambda(num_terms, size, numerator, divisor):
    vk = np.array([0] * num_terms)
    while True:
        x_0 = np.random.randint(0, 2, size=size)
        obj_1, uk, obj_2, _ = objective_value(x_0, numerator, divisor, vk)
        if (obj_2 >= 0) and check_if_neg(uk) and np.isfinite(obj_1):
            break
    v = vk
    u = np.array(uk)
    lamb = (v + u) / 2
    return lamb, u, v, uk, vk, x_0


def update_lambda(lamb, u, v, uk, vk, obj_2, sub_obj_1, sub_obj_2):
    if min(sub_obj_2) < 0:
        for t in range(len(sub_obj_2)):
            if sub_obj_2[t] < 0:
                lamb[t] = sub_obj_1[t]
        v = vk
        ta = uk / max(lamb)
        u = ta * lamb
    elif obj_2 > 0:
        v = lamb
    elif obj_2 < 0:
        u = lamb
    lamb = (v + u) / 2
    return lamb, u, v
