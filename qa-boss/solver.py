import numpy as np
import neal
import itertools

import pandas as pd


def sa_solver(bqm, previous_solution, num_reads=100):
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads)
    current_solution = previous_solution
    for sample, energy in response.data(['sample', 'energy']):
        if sum(sample.values()) > 0:
            current_solution = sample
    return current_solution


def exact_solver(numerator, divisor, size, num_terms):
    objective_value, objective_solution = float('Inf'), float('Inf')
    comb = [np.array(i) for i in itertools.product([0, 1], repeat=size)]
    for i in range(len(comb)):
        total_value = 0
        for n in range(num_terms):
            num = np.sum(comb[i] * numerator[n])
            den = np.sum(comb[i] * divisor[n])
            if den != 0:
                val = num / den
            else:
                val = float('Inf')
            total_value += val
        if objective_value > total_value:
            objective_value = total_value
            objective_solution = comb[i]
    return objective_value, objective_solution
