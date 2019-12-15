import numpy as np
import neal
import itertools


# from dwave.system.samplers import DWaveSampler
# from dwave.system.composites import EmbeddingComposite
from dwave.system import EmbeddingComposite, DWaveSampler
import dinkelbach as dkb


def sa_solver(bqm, previous_solution, num_reads=100):
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads)
    current_solution = previous_solution
    for sample, energy in response.data(['sample', 'energy']):
        current_solution = sample
        size = len(previous_solution)
        solution = []
        alphabet = list(current_solution.keys())[0][0]
        for i in range(size - 1):
            key = alphabet+'[{}]'.format(i)
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
            alphabet = list(current_solution.keys())[0][0]
            for i in range(size - 1):
                key = alphabet + '[{}]'.format(i)
                solution.append(current_solution[key])
            current_solution = np.append(np.array(solution),[1])
    return current_solution,response



def exact_solver(numerator, divisor, size, num_terms, is_spin = 0):
    if is_spin == 1:
        binary = [-1,1]
    else:
        binary = [0,1]
    min_value, objective_solution = float('Inf'), []
    comb = [np.append(np.array(i), [1]) for i in itertools.product(binary, repeat=size)]
    keep = 0
    for i in range(len(comb)):
        obj_1, sub, _, _, n, d = dkb.objective_value(comb[i], numerator, divisor, np.array([1]*num_terms))
        if min_value > obj_1:
            min_value = obj_1
            objective_solution = comb[i]
            keep = sub
    return min_value, objective_solution, keep



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