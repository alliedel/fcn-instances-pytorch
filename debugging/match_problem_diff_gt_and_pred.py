import numpy as np
import torch
from ortools.graph import pywrapgraph

from instanceseg.losses import match
from scipy import optimize


def solve_matching_with_scipy(cost_matrix_np):
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix_np)
    return row_ind, col_ind


def solve_matching_problem(cost_matrix, multiplier_for_db_print=1.0):
    assignment = pywrapgraph.LinearSumAssignment()
    # print('APD: Cost matrix size: {}'.format((len(cost_matrix), len(cost_matrix[0]))))
    for prediction in range(len(cost_matrix)):
        for ground_truth in range(len(cost_matrix[0])):
            try:
                assignment.AddArcWithCost(ground_truth, prediction,
                                          cost_matrix[prediction][ground_truth])
            except:
                print(cost_matrix[prediction][ground_truth])
                import ipdb; ipdb.set_trace()
                raise
    match.check_status(assignment.Solve(), assignment)
    match.debug_print_assignments(assignment, multiplier_for_db_print)
    return assignment

import time
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))


def main():
    # cost_matrix = [[cost(w, t) for w in workers] for t in tasks]
    n_workers = 100
    n_tasks = 100
    cost_matrix_np = np.ones((n_workers, n_tasks))
    for w, t, in enumerate(np.random.permutation(range(100))):
        cost_matrix_np[w, t] = 0
    with Timer('scipy torch'):
        ws, ts = solve_matching_with_scipy(torch.from_numpy(cost_matrix_np))
    with Timer('scipy numpy'):
        ws, ts = solve_matching_with_scipy(cost_matrix_np)
    cost_matrix_torch = [[torch.Tensor([cost_matrix_np[w, t]]) for t in range(n_tasks)] for w in range(n_workers)]
    cost_matrix, multiplier = match.convert_pytorch_costs_to_ints(cost_matrix_torch, infinity_cap=1e10)
    with Timer('google_opt'):
        assignment = solve_matching_problem(cost_matrix)

    ts = list(range(n_tasks))
    ws = [assignment.RightMate(t) for t in ts]
    print('APD: past pred_permutations')

    # for w, t in zip(ws, ts):
    #     print('cost(w={w},t={t}) = {cost}'.format(w=w, t=t, cost=float(cost_matrix_np[w, t])))

    return None


if __name__ == '__main__':
    assignment = main()
