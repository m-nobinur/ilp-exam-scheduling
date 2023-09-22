"""Greedy colouring for a feasible start, simulated annealing with Kempe-chain moves to improve it."""
import math
import random
from itertools import count
from time import perf_counter

import numpy as np

import model

W = np.array([0, 16, 8, 4, 2, 1])


def greedy(C, order="dsatur"):
    """Colour the conflict graph. order is 'largest_first' (Welsh-Powell) or 'dsatur' (Brelaz)."""
    n = len(C)
    adj = [np.flatnonzero(C[i]) for i in range(n)]
    deg = [len(a) for a in adj]
    colour = [-1] * n

    def first_free(i):
        used = {colour[j] for j in adj[i]}
        return next(c for c in count() if c not in used)

    if order == "largest_first":
        for i in sorted(range(n), key=lambda i: (-deg[i], i)):
            colour[i] = first_free(i)
        return colour
    sat = [set() for _ in range(n)]
    for _ in range(n):
        i = max((k for k in range(n) if colour[k] < 0), key=lambda k: (len(sat[k]), deg[k], -k))
        colour[i] = first_free(i)
        for j in adj[i]:
            sat[j].add(colour[i])
    return colour


def initial(inst, T):
    """Best greedy colouring that fits in T periods, else a feasibility solve of the model."""
    C = inst.matrix()
    for order in ("dsatur", "largest_first"):
        colour = greedy(C, order)
        if max(colour) < T:
            return colour, order
    _, timetable = model.solve(model.build_compact(inst, T, objective=False), time_limit=300)
    return timetable, "ilp-feasibility"


class Annealer:
    """Keeps L[i, q] = students shared by exam i and the exams now in period q."""

    def __init__(self, C, T, timetable):
        self.C, self.T = C, T
        gap = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
        self.Wp = np.where(gap <= 5, W[np.minimum(gap, 5)], 0)
        self.p = np.array(timetable)
        self.L = C @ np.eye(T, dtype=np.int64)[self.p]
        self.adj = [np.flatnonzero(C[i]) for i in range(len(C))]

    def total(self):
        return int(np.einsum("it,it->", self.Wp[self.p], self.L)) // 2

    def move(self, i, q):
        self.L[:, self.p[i]] -= self.C[:, i]
        self.L[:, q] += self.C[:, i]
        self.p[i] = q

    def chain(self, i, b):
        """Kempe chain of exam i between its period and period b."""
        a, seen, stack = self.p[i], {i}, [i]
        while stack:
            k = stack.pop()
            for j in self.adj[k]:
                if j not in seen and self.p[j] in (a, b):
                    seen.add(j)
                    stack.append(j)
        return seen

    def swap(self, chain, a, b):
        old = {k: self.p[k] for k in chain}
        for k in chain:
            self.move(k, b if old[k] == a else a)


def anneal(C, T, start, seconds, seed, cooling_to=1e-4):
    """Simulated annealing over Kempe-chain moves.
    A single-exam move is a chain of length one."""
    rng = random.Random(seed)
    sa = Annealer(C, T, start)
    n, cost = len(C), sa.total()
    best, best_p = cost, sa.p.copy()

    worse = [d for d in ((sa.Wp[q] - sa.Wp[sa.p[i]]) @ sa.L[i] for i, q in
             ((rng.randrange(n), rng.randrange(T)) for _ in range(200))) if d > 0]
    t0 = float(np.mean(worse)) if worse else 1.0
    t_end = t0 * cooling_to
    temp, moves, t_start = t0, 0, perf_counter()
    while True:
        if moves % 500 == 0:
            frac = (perf_counter() - t_start) / seconds
            if frac >= 1:
                break
            temp = t0 * (t_end / t0) ** frac
        moves += 1
        i, b = rng.randrange(n), rng.randrange(T)
        a = sa.p[i]
        if a == b:
            continue
        if sa.L[i, b] == 0:
            delta = int((sa.Wp[b] - sa.Wp[a]) @ sa.L[i])
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                sa.move(i, b)
                cost += delta
        else:
            chain = sa.chain(i, b)
            sa.swap(chain, a, b)
            delta = sa.total() - cost
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                cost += delta
            else:
                sa.swap(chain, a, b)
        if cost < best:
            best, best_p = cost, sa.p.copy()
    return best_p.tolist(), best, moves
