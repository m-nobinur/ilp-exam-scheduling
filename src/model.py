"""Compact and pairwise binary programs, built in PuLP and solved by HiGHS (highspy) or CBC."""
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import highspy
import pulp

W = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1}


@dataclass
class Model:
    prob: pulp.LpProblem
    formulation: str
    inst: object
    T: int
    offset: int = 0

    def size(self):
        return len(self.prob.variables()), len(self.prob.constraints)


def _assignment(prob, inst, T, free, fixed):
    x = {(i, p): pulp.LpVariable(f"x_{i}_{p}", cat="Binary") for i in free for p in range(T)}
    for i in free:
        prob += pulp.lpSum(x[i, p] for p in range(T)) == 1, f"assign_{i}"
    for (i, j), c in inst.conflicts.items():
        if i in free and j in free:
            for p in range(T):
                prob += x[i, p] + x[j, p] <= 1, f"clash_{i}_{j}_{p}"
        elif i in free or j in free:
            k, q = (i, fixed[j]) if i in free else (j, fixed[i])
            x[k, q].upBound = 0
    return x


def _fixed_cost(inst, T, free, fixed, x):
    """Linear cost of pairs with one free exam, constant cost of pairs with none."""
    terms, offset = [], 0
    for (i, j), c in inst.conflicts.items():
        if i in free and j in free:
            continue
        if i in free or j in free:
            k, q = (i, fixed[j]) if i in free else (j, fixed[i])
            terms += [W[abs(p - q)] * c * x[k, p] for p in range(T) if abs(p - q) in W]
        else:
            offset += W.get(abs(fixed[i] - fixed[j]), 0) * c
    return terms, offset


def cliques(inst, min_size=3):
    """Greedy maximal cliques of the conflict graph, one grown from each exam."""
    adj = {i: set() for i in range(inst.n)}
    for i, j in inst.conflicts:
        adj[i].add(j)
        adj[j].add(i)
    found = set()
    for v in range(inst.n):
        K, cand = {v}, set(adj[v])
        while cand:
            u = max(cand, key=lambda u: (len(adj[u] & cand), -u))
            K.add(u)
            cand &= adj[u]
        if len(K) >= min_size:
            found.add(tuple(sorted(K)))
    return sorted(found)


def mirror_exam(inst):
    """Exam with the largest total conflict, kept in the first half of the periods."""
    return max(range(inst.n), key=lambda i: sum(c for pair, c in inst.conflicts.items() if i in pair))


def build_compact(inst, T, fixed=None, objective=True, strong=False):
    """One z per conflicting pair and gap. Exams in `fixed` (exam -> period) are
    constants. strong=True adds clique, clique-window and symmetry-breaking
    constraints (chapter 4)."""
    fixed = fixed or {}
    free = [i for i in range(inst.n) if i not in fixed]
    prob = pulp.LpProblem("compact", pulp.LpMinimize)
    x = _assignment(prob, inst, T, set(free), fixed)
    terms, offset = _fixed_cost(inst, T, set(free), fixed, x)
    gaps = [g for g in W if g < T]
    z = {}
    for (i, j), c in inst.conflicts.items():
        if i in fixed or j in fixed:
            continue
        for g in gaps:
            z[i, j, g] = pulp.LpVariable(f"z_{i}_{j}_{g}", lowBound=0, upBound=1)
            terms.append(W[g] * c * z[i, j, g])
            for p in range(T):
                near = [x[j, q] for q in (p - g, p + g) if 0 <= q < T]
                if near:
                    prob += (z[i, j, g] >= x[i, p] + pulp.lpSum(near) - 1,
                             f"gap_{i}_{j}_{g}_{p}")
    if strong and not fixed:
        for k, K in enumerate(cliques(inst)):
            for p in range(T):
                prob += pulp.lpSum(x[i, p] for i in K) <= 1, f"clique_{k}_{p}"
            for g in gaps:
                n_win = [min(p, T - 1 - g) - max(0, p - g) + 1 for p in range(T)]
                pairs = pulp.lpSum((g + 1 - h) * z[i, j, h] for i in K for j in K
                                   if i < j for h in range(1, g + 1))
                spread = pulp.lpSum(n_win[p] * x[i, p] for i in K for p in range(T))
                prob += pairs >= spread - (T - g), f"window_{k}_{g}"
        first_half = range((T + 1) // 2)
        prob += pulp.lpSum(x[mirror_exam(inst), p] for p in first_half) == 1, "mirror"
    if objective:
        prob += pulp.lpSum(terms)
    return Model(prob, "strong" if strong and not fixed else "compact", inst, T, offset)


def build_pairwise(inst, T):
    """Textbook linearisation: one y per conflicting pair and ordered pair of periods within 5."""
    prob = pulp.LpProblem("pairwise", pulp.LpMinimize)
    x = _assignment(prob, inst, T, set(range(inst.n)), {})
    terms = []
    for (i, j), c in inst.conflicts.items():
        for p in range(T):
            for q in range(T):
                if abs(p - q) in W:
                    y = pulp.LpVariable(f"y_{i}_{j}_{p}_{q}", lowBound=0, upBound=1)
                    terms.append(W[abs(p - q)] * c * y)
                    prob += y >= x[i, p] + x[j, q] - 1, f"pair_{i}_{j}_{p}_{q}"
    prob += pulp.lpSum(terms)
    return Model(prob, "pairwise", inst, T)


def start_values(model, timetable):
    """Full solution vector (x, z or y) for a timetable, keyed by column name."""
    if model.formulation == "strong" and timetable[mirror_exam(model.inst)] >= (model.T + 1) // 2:
        timetable = [model.T - 1 - p for p in timetable]
    values = {}
    for v in model.prob.variables():
        kind, *idx = v.name.split("_")
        idx = list(map(int, idx))
        if kind == "x":
            values[v.name] = float(timetable[idx[0]] == idx[1])
        elif kind == "z":
            values[v.name] = float(abs(timetable[idx[0]] - timetable[idx[1]]) == idx[2])
        else:
            values[v.name] = float(timetable[idx[0]] == idx[2] and timetable[idx[1]] == idx[3])
    return values


def _highs(mps, time_limit, start, relax):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.readModel(str(mps))
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 0.0)
    h.setOptionValue("random_seed", 0)
    if relax:
        h.setOptionValue("solver", "ipm")
    names = list(h.getLp().col_names_)
    if start:
        sol = Path(mps).with_suffix(".start")
        rows = ["Model status", "None", "", "# Primal solution values", "Feasible", "Objective 0", f"# Columns {len(names)}"]
        sol.write_text("\n".join(rows + [f"{n} {start.get(n, 0.0)}" for n in names]) + "\n")
        h.readSolution(str(sol), 0)
    t0 = perf_counter()
    h.run()
    elapsed = perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    found = info.primal_solution_status == 2
    values = dict(zip(names, h.getSolution().col_value)) if found else {}
    return {"status": status, "optimal": status == "Optimal",
            "objective": info.objective_function_value if found else None,
            "bound": info.objective_function_value if relax else info.mip_dual_bound,
            "nodes": info.mip_node_count, "time": elapsed}, values


def _cbc(mps, time_limit):
    sol = Path(mps).with_suffix(".sol")
    cmd = ["cbc", str(mps), "timeMode", "elapsed", "sec", str(time_limit), "threads", "1",
           "ratioGap", "0", "allowableGap", "0", "solve", "solu", str(sol)]
    t0 = perf_counter()
    log = subprocess.run(cmd, capture_output=True, text=True).stdout
    elapsed = perf_counter() - t0

    def grab(pattern):
        m = re.search(pattern, log)
        return float(m.group(1)) if m else None

    result = re.search(r"Result - (.+)", log).group(1).strip()
    status = {"Optimal solution found": "Optimal", "Stopped on time limit": "Time limit reached"}.get(result, result)
    values = {}
    if sol.exists():
        lines = sol.read_text().splitlines()
        if "objective value" in lines[0]:
            for line in lines[1:]:
                parts = line.replace("**", "").split()
                values[parts[1]] = float(parts[2])
    objective = grab(r"Objective value:\s+(\S+)") if values else None
    bound = grab(r"Lower bound:\s+(\S+)")
    if status == "Optimal":
        bound = objective
    return {"status": status, "optimal": status == "Optimal", "objective": objective,
            "bound": bound, "nodes": int(grab(r"Enumerated nodes:\s+(\d+)") or 0), "time": elapsed}, values


def solve(model, solver="highs", time_limit=120, start=None, relax=False):
    """Solve and return (result dict, timetable as list of periods or None). Costs are per student."""
    if relax:
        for v in model.prob.variables():
            v.cat = pulp.LpContinuous
    with tempfile.TemporaryDirectory() as tmp:
        mps = Path(tmp) / "model.mps"
        model.prob.writeMPS(str(mps))
        if solver == "highs":
            res, values = _highs(mps, time_limit, start, relax)
        else:
            res, values = _cbc(mps, time_limit)
    S = len(model.inst.students)
    for key in ("objective", "bound"):
        if res[key] is not None and abs(res[key]) != float("inf"):
            res[key] = (res[key] + model.offset) / S
        else:
            res[key] = None
    obj, bound = res["objective"], res["bound"]
    res["gap"] = (obj - bound) / obj if obj and bound is not None else (0.0 if obj == 0 else None)
    n_vars, n_cons = model.size()
    res.update(formulation=model.formulation, solver=solver, periods=model.T, time_limit=time_limit,
               variables=n_vars, constraints=n_cons)
    timetable = None
    if values and not relax:
        timetable = [None] * model.inst.n
        for name, v in values.items():
            if name.startswith("x_") and v > 0.5:
                i, p = map(int, name.split("_")[1:])
                timetable[i] = p
    return res, timetable
