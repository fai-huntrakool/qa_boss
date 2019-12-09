import numpy as np
import neal
import itertools


# from dwave.system.samplers import DWaveSampler
# from dwave.system.composites import EmbeddingComposite
from dwave.system import EmbeddingComposite, DWaveSampler


def sa_solver(bqm, previous_solution, num_reads=100):
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads)
    current_solution = previous_solution
    for sample, energy in response.data(['sample', 'energy']):
        current_solution = sample
        size = len(previous_solution)
        solution = []
        for i in range(size - 1):
            key = 'x[{}]'.format(i)
            solution.append(current_solution[key])
        current_solution = np.append(np.array(solution), [1])
    return current_solution, response


def qa_solver(bqm, previous_solution, num_reads=100):
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample(bqm, num_reads=num_reads)
    current_solution = previous_solution

    ener = []
    for sample, energy in response.data(['sample', 'energy']):
        ener.append(energy)

    for i in ener:
        if round(i) == 0.0:
            condition = 0
            break
        else:
            condition = min(ener)

    for sample, energy in response.data(['sample', 'energy']):
        if energy == condition:
            current_solution = sample
            size = len(previous_solution)
            solution = []
            for i in range(size - 1):
                key = 'x[{}]'.format(i)
                solution.append(current_solution[key])
            current_solution = np.append(np.array(solution),[1])
    return current_solution,response



def exact_solver(numerator, divisor, size, num_terms):
    objective_value, objective_solution = float('Inf'), float('Inf')
    comb = [np.array(i) for i in itertools.product([0, 1], repeat=size)]
    lamb = []
    for i in range(len(comb)):
        total_value = 0
        keep_lamb = []
        for n in range(num_terms):
            num = np.sum(comb[i] * numerator[n][:-1]) + numerator[n][-1]
            den = np.sum(comb[i] * divisor[n][:-1]) + divisor[n][-1]
            val = num / den
            total_value += val
            keep_lamb.append(val)
        if objective_value > total_value:
            objective_value = total_value
            objective_solution = comb[i]
            ideal_lamb = keep_lamb
    return objective_value, objective_solution, ideal_lamb



#
# def sa_solver(bqm, previous_solution, num_reads=100):
#     sampler = neal.SimulatedAnnealingSampler()
#     response = sampler.sample(bqm, num_reads=num_reads)
#     current_solution = previous_solution
#     for sample, energy in response.data(['sample', 'energy']):
#         if sum(sample.values()) > 0:
#             current_solution = sample
#             size = len(previous_solution)
#             solution = []
#             for i in range(size-1):
#                 key = 'x[{}]'.format(i)
#                 solution.append(current_solution[key])
#             current_solution = np.append(np.array(solution), [1])
#     return current_solution