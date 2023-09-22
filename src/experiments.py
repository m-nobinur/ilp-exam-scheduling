"""Experiments E1-E6 and the synthetic department example. Every run is saved as one JSON file
and skipped when that file already exists.

usage: python -I src/experiments.py [pulp e1 e2 lp e3 e4 e5 e6 dept]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import data
import heuristic
import model
from evaluate import evaluate

RUNS = data.ROOT / "outputs" / "runs"
BENCHMARKS = ["hec-s-92", "sta-f-83"]
E2_SIZES = [6, 8, 10, 15, 20, 25, 30, 40]
SERIES = [("compact", "highs"), ("compact", "cbc"), ("pairwise", "highs"), ("pairwise", "cbc"), ("strong", "highs")]


def path(tag):
    return RUNS / f"{tag}.json"


def save(tag, record):
    RUNS.mkdir(parents=True, exist_ok=True)
    path(tag).write_text(json.dumps({"run": tag, **record}, indent=1, default=float))
    print(f"saved {tag}", flush=True)
    return record


def load_run(tag):
    return json.loads(path(tag).read_text()) if path(tag).exists() else None


def scored(inst, timetable):
    feasible, cost, total = evaluate(timetable, inst.students)
    return {"feasible": feasible, "evaluator_cost": cost, "evaluator_total": total}


def run_model(tag, inst, formulation, solver, T, time_limit, start=None, relax=False):
    if path(tag).exists():
        return load_run(tag)
    t0 = perf_counter()
    m = model.build_pairwise(inst, T) if formulation == "pairwise" else \
        model.build_compact(inst, T, strong=formulation == "strong")
    build_time = perf_counter() - t0
    res, tt = model.solve(m, solver, time_limit, model.start_values(m, start) if start else None, relax)
    record = {"instance": inst.name, "exams": inst.n, "students": len(inst.students),
              "pairs": len(inst.conflicts), "relaxed": relax, "build_time": build_time, **res, "timetable": tt}
    if tt:
        record.update(scored(inst, tt))
        assert record["feasible"] and record["evaluator_cost"] <= res["objective"] + 1e-6, tag
        assert not res["optimal"] or abs(record["evaluator_cost"] - res["objective"]) < 1e-6, tag
        record["solver_objective"], record["objective"] = res["objective"], record["evaluator_cost"]
        bound = res["bound"]
        if bound is not None and record["objective"]:
            record["gap"] = (record["objective"] - bound) / record["objective"]
    return save(tag, record)


def e1_instance():
    exams = list("ABCDEF")
    groups = [("AB", 3), ("AC", 2), ("BCD", 2), ("DE", 4), ("EF", 3), ("AF", 1), ("CE", 2), ("BF", 2), ("AD", 1)]
    students = [tuple(exams.index(e) for e in g) for g, k in groups for _ in range(k)]
    return data.Instance("e1-toy", exams, students, 4)


def e1():
    inst = e1_instance()
    if not path("e1-bruteforce").exists():
        t0 = perf_counter()
        costs = [(evaluate(tt, inst.students), tt) for tt in product(range(inst.periods), repeat=inst.n)]
        feasible = [(c[1], tt) for c, tt in costs if c[0]]
        best = min(c for c, _ in feasible)
        save("e1-bruteforce", {"instance": inst.name, "exams": inst.n, "students": len(inst.students),
                               "periods": inst.periods, "timetables": len(costs), "feasible_timetables": len(feasible),
                               "optimal_timetables": sum(c == best for c, _ in feasible),
                               "objective": best, "time": perf_counter() - t0,
                               "timetable": list(next(tt for c, tt in feasible if c == best))})
    values = [load_run("e1-bruteforce")["objective"]]
    for f, s in SERIES:
        values.append(run_model(f"e1-{f}-{s}", inst, f, s, inst.periods, 60)["objective"])
    for f in ("compact", "strong"):
        run_model(f"e1-{f}-lp", inst, f, "highs", inst.periods, 60, relax=True)
    assert max(values) - min(values) < 1e-9, values
    print("E1 agree:", values)


def e2_periods(inst):
    return max(heuristic.greedy(inst.matrix(), "dsatur")) + 3


def e2_series(formulation, solver):
    model.solve(model.build_compact(e1_instance(), 4), solver)
    full = data.load("hec-s-92")
    for n in E2_SIZES:
        inst = data.sub_instance(full, n)
        rec = run_model(f"e2-{formulation}-{solver}-n{n}", inst, formulation, solver, e2_periods(inst), 120)
        if not rec["optimal"]:
            break


def e2():
    with ProcessPoolExecutor(len(SERIES)) as pool:
        list(pool.map(e2_series, *zip(*SERIES)))
    for n in E2_SIZES:
        opt = [r["objective"] for f, s in SERIES if (r := load_run(f"e2-{f}-{s}-n{n}")) and r["optimal"]]
        assert not opt or max(opt) - min(opt) < 1e-6, (n, opt)


def lp():
    full = data.load("hec-s-92")
    subs = [data.sub_instance(full, n) for n in E2_SIZES]
    for inst in [data.load(name) for name in BENCHMARKS] + subs:
        T = inst.periods or e2_periods(inst)
        for f in ("compact", "strong"):
            run_model(f"lp-{inst.name}-{f}", inst, f, "highs", T, 300, relax=True)


def e3_seed(name, seed, seconds=60):
    tag = f"e3-{name}-sa-seed{seed}"
    if path(tag).exists():
        return
    inst = data.load(name)
    start = load_run(f"e3-{name}-start")["timetable"]
    tt, total, moves = heuristic.anneal(inst.matrix(), inst.periods, start, seconds, seed)
    save(tag, {"instance": name, "seed": seed, "time_limit": seconds, "moves": moves,
               "objective": total / len(inst.students), "timetable": tt, **scored(inst, tt)})


def e3():
    for name in BENCHMARKS:
        tag = f"e3-{name}-start"
        if not path(tag).exists():
            inst = data.load(name)
            C = inst.matrix()
            colours = {o: max(heuristic.greedy(C, o)) + 1 for o in ("dsatur", "largest_first")}
            t0 = perf_counter()
            tt, how = heuristic.initial(inst, inst.periods)
            save(tag, {"instance": name, "periods": inst.periods, "colours": colours, "method": how,
                       "time": perf_counter() - t0, "objective": evaluate(tt, inst.students)[1],
                       "timetable": tt, **scored(inst, tt)})
    jobs = [(name, seed) for name in BENCHMARKS for seed in range(5)]
    with ProcessPoolExecutor(len(jobs)) as pool:
        list(pool.map(e3_seed, *zip(*jobs)))


def best_e3(name):
    return min((load_run(f"e3-{name}-sa-seed{s}") for s in range(5)), key=lambda r: r["objective"])


def e4_one(name, formulation):
    inst = data.load(name)
    warm = best_e3(name)
    rec = run_model(f"e4-{name}-{formulation}-highs-warm", inst, formulation, "highs", inst.periods, 480,
                    warm["timetable"])
    assert rec["objective"] <= warm["objective"] + 1e-9, "warm start was dropped"


def e4():
    jobs = [(name, f) for name in BENCHMARKS for f in ("compact", "strong")]
    with ProcessPoolExecutor(len(jobs)) as pool:
        list(pool.map(e4_one, *zip(*jobs)))


def fix_and_optimise(inst, T, timetable, budget=480, window=3, sub_limit=20):
    """Free the exams in a window of periods, fix the rest, solve exactly. A pass with no
    improvement widens the window by one period."""
    tt = list(timetable)
    cost = evaluate(tt, inst.students)[2]
    history, solves, proven, t0 = [(0.0, cost, window)], 0, 0, perf_counter()
    while window <= T and perf_counter() - t0 < budget:
        improved = False
        for s in range(T - window + 1):
            left = budget - (perf_counter() - t0)
            if left < 1:
                break
            free = {i for i in range(inst.n) if s <= tt[i] < s + window}
            m = model.build_compact(inst, T, fixed={i: tt[i] for i in range(inst.n) if i not in free})
            res, sub = model.solve(m, "highs", min(sub_limit, left), model.start_values(m, tt))
            solves, proven = solves + 1, proven + res["optimal"]
            if sub is None:
                continue
            new = [sub[i] if i in free else tt[i] for i in range(inst.n)]
            ok, _, new_cost = evaluate(new, inst.students)
            if ok and new_cost < cost:
                tt, cost, improved = new, new_cost, True
            history.append((perf_counter() - t0, cost, window))
        if not improved:
            window += 1
    return tt, cost, history, solves, proven


def e5_one(name):
    tag = f"e5-{name}-fix-optimise"
    if path(tag).exists():
        return
    inst = data.load(name)
    e4_runs = [load_run(f"e4-{name}-{f}-highs-warm") for f in ("compact", "strong")]
    start = min([best_e3(name)] + e4_runs, key=lambda r: r["objective"])
    tt, total, history, solves, proven = fix_and_optimise(inst, inst.periods, start["timetable"])
    S = len(inst.students)
    save(tag, {"instance": name, "start_run": start["run"], "start_objective": start["objective"],
               "objective": total / S, "subproblems": solves, "subproblems_optimal": proven,
               "time": history[-1][0], "widest_window": history[-1][2],
               "history": [(t, c / S, w) for t, c, w in history],
               "timetable": tt, **scored(inst, tt)})


def e5():
    with ProcessPoolExecutor(2) as pool:
        list(pool.map(e5_one, BENCHMARKS))


def e6_exact():
    full = data.load("hec-s-92")
    solved = [n for n in E2_SIZES if (r := load_run(f"e2-compact-highs-n{n}")) and r["optimal"]]
    inst = data.sub_instance(full, max(solved))
    T = e2_periods(inst) - 2
    while True:
        tag = f"e6-feasible-n{inst.n}-T{T - 1}"
        if not path(tag).exists():
            res, tt = model.solve(model.build_compact(inst, T - 1, objective=False), "highs", 120)
            save(tag, {"instance": inst.name, "periods": T - 1, **res, "timetable": tt})
        if load_run(tag)["timetable"] is None:
            break
        T -= 1
    for t in range(T, T + 7):
        run_model(f"e6-exact-n{inst.n}-T{t}", inst, "compact", "highs", t, 120)


def e6_heuristic(T, seed, seconds=30):
    tag = f"e6-hec-s-92-T{T}-seed{seed}"
    if path(tag).exists():
        return
    inst = data.load("hec-s-92")
    start, how = heuristic.initial(inst, T)
    tt, total, _ = heuristic.anneal(inst.matrix(), T, start, seconds, seed)
    save(tag, {"instance": inst.name, "periods": T, "seed": seed, "start": how, "time_limit": seconds,
               "objective": total / len(inst.students), "timetable": tt, **scored(inst, tt)})


def e6():
    jobs = [(T, seed) for T in range(18, 25) for seed in range(3)]
    with ProcessPoolExecutor(7) as pool:
        heavy = pool.submit(e6_exact)
        list(pool.map(e6_heuristic, *zip(*jobs)))
        heavy.result()


def synthetic_department(path_out, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    programmes = {f"P{k}": [f"P{k}-{y}{m}" for y in (1, 2) for m in "AB"] for k in range(1, 5)}
    electives = [f"EL-{k}" for k in range(1, 9)]
    for s in range(300):
        prog = programmes[f"P{rng.integers(1, 5)}"]
        year = rng.integers(1, 3)
        exams = [e for e in prog if e.split("-")[1][0] == str(year)]
        exams += list(rng.choice(electives, size=rng.integers(1, 3), replace=False))
        rows += [(f"S{s:04d}", e) for e in exams]
    pd.DataFrame(rows, columns=["student_id", "exam_code"]).to_csv(path_out, index=False)


def dept():
    folder = data.ROOT / "data"
    template = folder / "department_template.csv"
    if not template.exists():
        template.write_text("student_id,exam_code\n")
    csv = folder / "synthetic_department.csv"
    if not csv.exists():
        synthetic_department(csv)
    inst = data.from_csv(csv, "synthetic-department")
    T = max(heuristic.greedy(inst.matrix(), "dsatur")) + 3
    exact = run_model("dept-synthetic-strong-highs", inst, "strong", "highs", T, 120)
    if not path("dept-synthetic-sa").exists():
        start, how = heuristic.initial(inst, T)
        tt, total, moves = heuristic.anneal(inst.matrix(), T, start, 30, 0)
        save("dept-synthetic-sa", {"instance": inst.name, "periods": T, "time_limit": 30, "moves": moves,
                                   "objective": total / len(inst.students), "timetable": tt, **scored(inst, tt)})
    assert load_run("dept-synthetic-sa")["objective"] >= exact["bound"] - 1e-9


def pulp_status():
    import re, shutil, pulp
    if path("pulp-status-hec-s-92").exists():
        return
    inst = data.sub_instance(data.load("hec-s-92"), 15)
    m = model.build_compact(inst, 18)
    log = RUNS.parent / "logs" / "pulp-status.log"
    m.prob.solve(pulp.COIN_CMD(path=shutil.which("cbc"), timeLimit=20, threads=1, logPath=str(log)))
    save("pulp-status-hec-s-92", {"instance": inst.name, "exams": inst.n, "time_limit": 20,
                                  "pulp_status": pulp.LpStatus[m.prob.status],
                                  "pulp_solution_status": pulp.LpSolution[m.prob.sol_status],
                                  "cbc_status": re.search(r"Result - (.+)", log.read_text()).group(1).strip()})


STAGES = {"pulp": pulp_status, "e1": e1, "e2": e2, "lp": lp, "e3": e3, "e4": e4, "e5": e5, "e6": e6, "dept": dept}

if __name__ == "__main__":
    for stage in sys.argv[1:] or STAGES:
        print(f"== {stage}", flush=True)
        STAGES[stage]()
