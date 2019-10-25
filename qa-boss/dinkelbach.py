import numpy as np


def objective_value(x, numerator, divisor, lamb):
    # obj_1 = num/den
    # obj_2 = num - lamb*den

    num = np.dot(numerator, x)
    den = np.dot(divisor, x)
    den_with_lamb = np.dot(lamb, den)

    sub_obj_1 = [num[i] / den[i] if den[i] != 0 else float('Inf') for i in range(len(num))]
    obj_1 = np.sum(sub_obj_1)

    sub_obj_2 = [num[i] - den_with_lamb[i] for i in range(len(num))]
    obj_2 = np.sum(sub_obj_2)
    return obj_1, sub_obj_1, obj_2, sub_obj_2
