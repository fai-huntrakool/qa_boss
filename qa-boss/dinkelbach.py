import numpy as np
import generate_test_prob as gtp
import solver
import numpy as np
from scipy.optimize import line_search


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
    return obj_1, sub_obj_1, obj_2, sub_obj_2, num, den


def objective_value_for_find_lamb(x, numerator, divisor):
    # obj_1 = num/den
    # obj_2 = num - lamb*den
    num = np.dot(numerator, x)
    den = np.dot(divisor, x)

    sub_obj_1 = [num[i] / den[i] if den[i] != 0 else float('Inf') for i in range(len(num))]
    obj_1 = np.sum(sub_obj_1)
    lamb = np.array(sub_obj_1) / 2
    den_with_lamb = lamb * den
    sub_obj_2 = [num[i] - den_with_lamb[i] for i in range(len(num))]
    obj_2 = np.sum(sub_obj_2)
    return obj_1, sub_obj_1, obj_2, sub_obj_2


def print_iteration_value(i, current_solution, obj_1, obj_2, lamb):
    print('iteration: {}'.format(i))
    print('solution: {}'.format(current_solution[:-1]))
    print('min P(x)-a*Q(X) = {}.'.format(obj_2))
    print('fix lambda={} : min P(x)/Q(X) = {}.'.format(lamb, obj_1))


def dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution, i=1):
    bqm = gtp.construct_bqm(x, lamb, numerator, divisor)
    current_solution = solver.sa_solver(bqm, previous_solution)
    obj_1, sub_obj_1, obj_2, _, _, _ = objective_value(current_solution, numerator, divisor, lamb)
    previous_solution = current_solution
    print_iteration_value(i, current_solution, obj_1, obj_2, lamb)
    print('--------------------------------------------')
    if obj_2 != 0:
        lamb = np.array(sub_obj_1)
        dinkelbach_for_one_ratio(x, lamb, numerator, divisor, previous_solution, i + 1)
    else:
        return x, obj_1, obj_2


def dinkelbach_for_multiple_ratios(x, lamb, u, v, uk, vk, numerator, divisor, previous_solution, limit_iteration, i=1):
    bqm = gtp.construct_bqm(x, lamb, numerator, divisor)
    current_solution = solver.sa_solver(bqm, previous_solution)
    obj_1, sub_obj_1, obj_2, sub_obj_2, num, den = objective_value(current_solution, numerator, divisor, lamb)
    previous_solution = current_solution
    print_iteration_value(i, current_solution, obj_1, obj_2, lamb)
    print('sub_obj_1 : {}'.format(sub_obj_1))
    print('sub_obj_2 : {}'.format(sub_obj_2))
    # print('u : {}'.format(u))
    # print('v : {}'.format(v))
    print('--------------------------------------------')
    if ((abs(obj_2) <= 0.005) and min(sub_obj_2) >= -0.005) | (i > limit_iteration):
        return x, obj_1, obj_2
    else:
        # lamb, u, v = update_lambda(lamb, u, v, uk, vk, obj_2, sub_obj_1, sub_obj_2)
        lamb = update_lambda_ls(lamb, num, den)
        dinkelbach_for_multiple_ratios(x, lamb, u, v, uk, vk, numerator, divisor, previous_solution, limit_iteration,
                                       i + 1)

#
#
# def update_lambda_ls(lamb, num, den):
#     delta = 100
#     while np.linalg.norm(delta) > 0.01:
#         f_value, func = f(lamb, num, den)
#         delta, d_func = delta_vector(lamb, num, den)
#         pt = descent_condition(delta)
#         alpha, _, _, _, _, _ = line_search(func, d_func, lamb, pt)
#         if alpha is None:
#             break
#         lamb += alpha*pt
#     return lamb

def update_lambda_ls(lamb, num, den):
    f_value, func = f(lamb, num, den)
    delta, d_func = delta_vector(lamb, num, den)
    pt = descent_condition(delta)*2
    alpha, _, _, _, _, _ = line_search(func, d_func, lamb, pt)
    i = 0
    while (alpha is None) and (i < 10):
        pt = descent_condition(delta)
        alpha, _, _, _, _, _ = line_search(func, d_func, lamb, pt)
    return lamb + alpha * pt


def check_if_neg(l):
    for i in l:
        if i < 0:
            return False
    return True


def delta_vector(lamb, num, den):
    size = len(num)
    delta = []
    for i in range(size):
        temp = -2 * num[i] * den[i] + 2 * den[i] * den[i] * lamb[i]
        delta.append(temp)
    return np.array(delta), lambda a: [-2 * num[i] * den[i] + 2 * den[i] * den[i] * a[i] for i in range(len(a))]


# def descent_condition(delta, pt):
#     p_k = delta
#     print(np.dot(pt, delta))
#     if -1 * np.dot(pt, delta) > 0:
#         return True
#     else:
#         return False
def descent_condition(delta):
    p_k = np.random.randint(-5, 5, len(delta))
    if np.linalg.norm(delta) == 0:
        return 0
    if np.dot(p_k, delta) / np.linalg.norm(p_k) / np.linalg.norm(delta) < 0:
        return p_k
    else:
        return descent_condition(delta)


def f(lamb, num, den):
    sum_term = 0
    for i in range(len(lamb)):
        sum_term += np.square(num[i] - lamb[i] * den[i])
    return sum_term, lambda a: np.sum([np.square(num[i] - a[i] * den[i]) for i in range(len(a))])


def gradient_related(m, delta, pt):
    if np.linalg.norm(pt) >= m * np.linalg.norm(delta):
        return True
    return False


def wolfe_condition(alpha, mu, pt, delta, lamb, num, den):
    if f(lamb + alpha * pt, num, den) <= f(lamb, num, den) + mu * alpha * pt * delta:
        return True
    return False


def initialize_lambda(num_terms, size, numerator, divisor):
    vk = np.array([0.0] * num_terms)
    while True:
        x_0 = np.append(np.random.randint(0, 2, size=size), [1])
        obj_1, uk, obj_2, sub_obj_2 = objective_value_for_find_lamb(x_0, numerator, divisor)
        if all(item >= 0 for item in sub_obj_2):
            break
    v = vk
    u = 2 * np.array(uk)
    lamb = (u + v) / 2
    return lamb, u, v, uk, vk, x_0, obj_1, obj_2


# Original Algorithm
# def update_lambda(lamb, u, v, uk, vk, obj_2, sub_obj_1, sub_obj_2):
#     if (abs(obj_2) <= 0.005) and (min(sub_obj_2) < 0.05):
#         for t in range(len(sub_obj_2)):
#             if sub_obj_2[t] < 0:
#                 lamb[t] = sub_obj_1[t]
#         v = vk
#         ta = uk / max(lamb)
#         u = ta * lamb
#
#     elif obj_2 > 0:
#         v = lamb
#         lamb = (v + u) / 2
#     elif obj_2 < 0:
#         u = lamb
#         lamb = (v + u) / 2
#     return lamb, u, v

# #Algorithm A
# def update_lambda(lamb, u, v, uk, vk, obj_2, sub_obj_1, sub_obj_2):
#     lamb = sub_obj_1
#     return lamb, u, v

#
# # Algorithm B
# def update_lambda(lamb, u, v, uk, vk, obj_2, sub_obj_1, sub_obj_2):
#     for i in range(len(sub_obj_2)):
#         if sub_obj_2[i] > 0:
#             u[i] = max(u[i], sub_obj_1[i])
#             v[i] = lamb[i]
#         else:
#             u[i] = lamb[i]
#             v[i] = min(v[i], sub_obj_1[i])
#     lamb = (v + u) / 2
#     return lamb, u, v


# Original Algorithm
# def update_lambda(lamb, u, v, uk, vk, obj_2, sub_obj_1, sub_obj_2):
#     if (abs(obj_2) <= 0.005) and (min(sub_obj_2) < 0.05):
#         for t in range(len(sub_obj_2)):
#             if sub_obj_2[t] < 0:
#                 lamb[t] = sub_obj_1[t]
#         v = vk
#         ta = uk / max(lamb)
#         u = ta * lamb
#
#     elif obj_2 > 0:
#         v = lamb
#         u = sub_obj_1
#         lamb = (v + u) / 2
#     elif obj_2 < 0:
#         u = lamb
#         v = sub_obj_1
#         lamb = (v + u) / 2
#     return lamb, u, v

def update_lambda(lamb, u, v, uk, vk, obj_2, sub_obj_1, sub_obj_2):
    for i in range(len(sub_obj_2)):
        if sub_obj_2[i] > 0:
            u[i] = (u[i] + sub_obj_1[i]) / 2
            v[i] = lamb[i]
        else:
            u[i] = lamb[i]
            v[i] = (v[i] + sub_obj_1[i]) / 2
    lamb = (v + u) / 2
    return lamb, u, v
