"""Tables, figures, best timetables, results.json and RESULTS.md, all from the saved run files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
import model
from evaluate import evaluate
from experiments import BENCHMARKS, E2_SIZES, RUNS, SERIES, best_e3, e2_periods, load_run

OUT = data.ROOT / "outputs"
TABLES, FIGS, TIMETABLES = OUT / "tables", OUT / "figures", OUT / "timetables"
PUBLISHED = {"hec-s-92": 10.03, "sta-f-83": 157.03}
STA_BEST = 95947 / 611

plt.rcParams.update({"font.family": ["Carlito", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
COLOUR = {"compact": "#2a78d6", "pairwise": "#eb6834", "strong": "#1baf7a"}
MARKER = {"compact": "o", "pairwise": "s", "strong": "^"}
DASH = {"highs": "-", "cbc": "--"}
NAME = {"compact": "Compact", "pairwise": "Pairwise", "strong": "Strengthened", "highs": "HiGHS", "cbc": "CBC"}

results, sources = {}, {}


def put(key, value, source, fmt="{:.2f}"):
    results[key] = fmt.format(value) if isinstance(value, (float, np.floating)) else str(value)
    sources[key] = source


def f2(v):
    return "" if v is None else f"{v:.2f}"


def pct(v):
    return "" if v is None else f"{100 * v:.1f}%"


def status(r):
    return "optimal" if r["optimal"] else ("time limit" if r["status"].startswith("Time") else r["status"].lower())


def table(name, rows):
    pd.DataFrame(rows).to_csv(TABLES / f"{name}.csv", index=False)


def save(fig, name):
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)


def instances_table():
    rows = []
    for name in BENCHMARKS:
        inst = data.load(name)
        st = inst.stats()
        C = inst.matrix()
        clique = max(map(len, model.cliques(inst)))
        start = load_run(f"e3-{name}-start")
        rows.append({"Instance": name, "Exams": st["exams"], "Students": f"{st['students']:,}",
                     "Enrolments": f"{st['enrolments']:,}", "Pairs": f"{st['conflicting_pairs']:,}",
                     "Density": f"{st['density']:.3f}", "Periods": st["periods"], "Clique": clique,
                     "DSatur": start["colours"]["dsatur"]})
        key = name.split("-")[0]
        for k in ("exams", "students", "enrolments", "conflicting_pairs", "periods"):
            put(f"{key}_{k}", f"{st[k]:,}", f"data/raw/{name}.*")
        put(f"{key}_density", st["density"], f"data/raw/{name}.*", "{:.3f}")
        put(f"{key}_density_nn1", 2 * st["conflicting_pairs"] / (st["exams"] * (st["exams"] - 1)),
            f"data/raw/{name}.*", "{:.4f}")
        put(f"{key}_clique", clique, f"data/raw/{name}.*")
        put(f"{key}_dsatur", start["colours"]["dsatur"], f"e3-{name}-start.json")
        put(f"{key}_lf", start["colours"]["largest_first"], f"e3-{name}-start.json")
        put(f"{key}_published", PUBLISHED[name], "Aslan 2023, Table 6")
        put(f"{key}_maxdeg", int((C > 0).sum(1).max()), f"data/raw/{name}.*")
        put(f"{key}_meandeg", float((C > 0).sum(1).mean()), f"data/raw/{name}.*", "{:.1f}")
    put("sta_best_known", STA_BEST, "best known total 95947 / 611 students")
    table("t_instances", rows)


def degree_figure():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=False)
    for ax, name in zip(axes, BENCHMARKS):
        deg = (data.load(name).matrix() > 0).sum(1)
        ax.hist(deg, bins=np.arange(0, deg.max() + 6, 5), color=COLOUR["compact"], edgecolor="white", linewidth=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Degree (exams in conflict)")
        ax.axvline(deg.mean(), color="#0b0b0b", linestyle="--", linewidth=1)
        ax.text(deg.mean(), ax.get_ylim()[1] * 0.92, f" mean {deg.mean():.1f}", fontsize=8)
    axes[0].set_ylabel("Number of exams")
    save(fig, "f_degree")


def e1_table():
    bf = load_run("e1-bruteforce")
    rows = [{"Method": "Brute force (evaluator)", "Cost": f2(bf["objective"]), "Status": "optimal",
             "Time (s)": f"{bf['time']:.3f}", "Nodes": f"{bf['timetables']:,} timetables"}]
    for f, s in SERIES:
        r = load_run(f"e1-{f}-{s}")
        rows.append({"Method": f"{NAME[f]}, {NAME[s]}", "Cost": f2(r["objective"]), "Status": status(r),
                     "Time (s)": f"{r['time']:.3f}", "Nodes": r["nodes"]})
    for f in ("compact", "strong"):
        r = load_run(f"e1-{f}-lp")
        rows.append({"Method": f"{NAME[f]}, LP relaxation", "Cost": f2(r["objective"]), "Status": "LP bound",
                     "Time (s)": f"{r['time']:.3f}", "Nodes": "-"})
        put(f"e1_lp_{f}", r["objective"], f"e1-{f}-lp.json")
    table("t_e1", rows)
    put("e1_opt", bf["objective"], "e1-bruteforce.json")
    put("e1_students", bf["students"], "e1-bruteforce.json")
    put("e1_total", f"{bf['timetables']:,}", "e1-bruteforce.json")
    put("e1_feasible", bf["feasible_timetables"], "e1-bruteforce.json")
    put("e1_optimal_count", bf["optimal_timetables"], "e1-bruteforce.json")


def pairwise_size(n, pairs, T):
    """Size of the pairwise model by formula (checked against the built model where a run exists)."""
    P = sum(1 for p in range(T) for q in range(T) if 1 <= abs(p - q) <= 5)
    return n * T + pairs * P, n + pairs * T + pairs * P


def e2_tables():
    full = data.load("hec-s-92")
    size_rows, rows = [], []
    for n in E2_SIZES:
        inst = data.sub_instance(full, n)
        T = e2_periods(inst)
        lpc, lps = load_run(f"lp-{inst.name}-compact"), load_run(f"lp-{inst.name}-strong")
        pv, pc = pairwise_size(n, len(inst.conflicts), T)
        if (r := load_run(f"e2-pairwise-highs-n{n}")):
            assert (r["variables"], r["constraints"]) == (pv, pc)
        size_rows.append({"n": n, "Students": f"{len(inst.students):,}", "Pairs": len(inst.conflicts), "T": T,
                          "Compact vars": f"{lpc['variables']:,}", "Compact cons": f"{lpc['constraints']:,}",
                          "Pairwise vars": f"{pv:,}", "Pairwise cons": f"{pc:,}",
                          "Strengthened cons": f"{lps['constraints']:,}"})
        for f, s in SERIES:
            r = load_run(f"e2-{f}-{s}-n{n}")
            if r:
                rows.append({"n": n, "T": T, "Model": NAME[f], "Solver": NAME[s], "Status": status(r),
                             "Cost": f2(r["objective"]), "Bound": f2(r["bound"]), "Gap": pct(r["gap"]),
                             "Time (s)": f"{r['time']:.1f}", "Nodes": f"{r['nodes']:,}"})
    table("t_e2_size", size_rows)
    table("t_e2", rows)
    for f, s in SERIES:
        solved = [n for n in E2_SIZES if (r := load_run(f"e2-{f}-{s}-n{n}")) and r["optimal"]]
        put(f"e2_largest_{f}_{s}", max(solved) if solved else "none", f"e2-{f}-{s}-n*.json")
        for n in E2_SIZES:
            if (r := load_run(f"e2-{f}-{s}-n{n}")):
                put(f"e2_time_{f}_{s}_n{n}", r["time"], f"e2-{f}-{s}-n{n}.json", "{:.1f}")
                put(f"e2_nodes_{f}_{s}_n{n}", f"{r['nodes']:,}", f"e2-{f}-{s}-n{n}.json")
    for n in E2_SIZES:
        r = load_run(f"e2-compact-highs-n{n}")
        if r:
            put(f"e2_cost_n{n}", r["objective"], f"e2-compact-highs-n{n}.json")
    return size_rows


def e2_figures(size_rows):
    df = pd.DataFrame(size_rows)
    num = lambda col: df[col].str.replace(",", "").astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for ax, what in zip(axes, ("vars", "cons")):
        for f in ("compact", "pairwise", "strong"):
            if f == "strong" and what == "vars":
                continue
            col = f"{NAME[f]} {what}"
            ax.plot(df["n"], num(col), marker=MARKER[f], color=COLOUR[f], linewidth=2, markersize=5, label=NAME[f])
        ax.set_yscale("log")
        ax.set_xlabel("Exams in sub-instance (n)")
        ax.set_title("Variables" if what == "vars" else "Constraints", fontsize=9)
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="lower right")
    save(fig, "f_model_size")

    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    for f, s in SERIES:
        runs = [load_run(f"e2-{f}-{s}-n{n}") for n in E2_SIZES]
        runs = [r for r in runs if r]
        n = [r["exams"] for r in runs]
        t = [r["time"] for r in runs]
        ax.plot(n, t, DASH[s], color=COLOUR[f], linewidth=1.6, marker=MARKER[f], markersize=6,
                label=f"{NAME[f]}, {NAME[s]}")
        for r in runs:
            if not r["optimal"]:
                ax.plot(r["exams"], r["time"], MARKER[f], markersize=6, color=COLOUR[f], markerfacecolor="white")
    ax.axhline(120, color="#52514e", linewidth=0.8, linestyle=":")
    ax.text(ax.get_xlim()[0], 135, "time limit 120 s", fontsize=8, color="#52514e")
    ax.set_yscale("log")
    ax.set_xlabel("Exams in sub-instance (n)")
    ax.set_ylabel("Solve time (s, log scale)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    save(fig, "f_solve_time")


def lp_table():
    full = data.load("hec-s-92")
    rows = []
    insts = [data.sub_instance(full, n) for n in E2_SIZES] + [data.load(b) for b in BENCHMARKS]
    for inst in insts:
        c, s = load_run(f"lp-{inst.name}-compact"), load_run(f"lp-{inst.name}-strong")
        if inst.name in PUBLISHED:
            ref, ref_src = PUBLISHED[inst.name], "best known (lit.)"
        else:
            r = load_run(f"e2-compact-highs-n{inst.n}")
            proven = [x for x in [r] + [load_run(f"e2-{f}-{sv}-n{inst.n}") for f, sv in SERIES] if x and x["optimal"]]
            ref, ref_src = (proven[0]["objective"], "optimum (E2)") if proven else (None, "")
        rows.append({"Instance": inst.name, "T": c["periods"], "Compact LP": f2(c["objective"]),
                     "Strengthened LP": f2(s["objective"]), "Reference cost": f2(ref), "Reference": ref_src})
        key = inst.name.replace("-", "_")
        put(f"lp_{key}_compact", c["objective"], c["run"] + ".json")
        put(f"lp_{key}_strong", s["objective"], s["run"] + ".json")
    table("t_lp", rows)


def benchmark_tables():
    rows, cost_rows = [], []
    for name in BENCHMARKS:
        inst, key = data.load(name), name.split("-")[0]
        start = load_run(f"e3-{name}-start")
        sa = [load_run(f"e3-{name}-sa-seed{s}") for s in range(5)]
        costs = np.array([r["objective"] for r in sa])
        label = {"dsatur": "DSatur", "largest_first": "Largest first", "ilp-feasibility": "Feasibility model"}
        rows.append({"Instance": name, "Start": label[start["method"]], "Start cost": f2(start["objective"]),
                     "Mean": f2(costs.mean()), "Std": f2(costs.std(ddof=1)), "Best": f2(costs.min()),
                     "Worst": f2(costs.max()), "Moves per run": f"{int(np.mean([r['moves'] for r in sa])):,}"})
        put(f"{key}_start", start["objective"], start["run"] + ".json")
        put(f"{key}_start_method", start["method"], start["run"] + ".json")
        put(f"{key}_sa_mean", costs.mean(), f"e3-{name}-sa-seed*.json")
        put(f"{key}_sa_std", costs.std(ddof=1), f"e3-{name}-sa-seed*.json")
        put(f"{key}_sa_best", costs.min(), best_e3(name)["run"] + ".json")
        put(f"{key}_sa_worst", costs.max(), f"e3-{name}-sa-seed*.json")
        hits = int(np.sum(np.isclose(costs, STA_BEST)))
        put(f"{key}_sa_hits", hits, f"e3-{name}-sa-seed*.json")
        put(f"{key}_sa_hits_text", ["none", "only one", "two", "three", "four", "all"][hits],
            f"e3-{name}-sa-seed*.json", "{}")
        put(f"{key}_sa_moves", f"{int(np.mean([r['moves'] for r in sa])):,}", f"e3-{name}-sa-seed*.json")
        put(f"{key}_sa_gap_published", 100 * (costs.min() - PUBLISHED[name]) / PUBLISHED[name],
            best_e3(name)["run"] + ".json", "{:.1f}")

        label = "Start (DSatur)" if start["method"] == "dsatur" else "Start (feasibility model)"
        cost_rows.append({"Instance": name, "Method": label, "Cost": f2(start["objective"]),
                          "Bound": "", "Gap": "", "Status": "feasible"})
        cost_rows.append({"Instance": name, "Method": "Simulated annealing (best of 5)", "Cost": f2(costs.min()),
                          "Bound": "", "Gap": "", "Status": "best found, 60 s"})
        for f in ("compact", "strong"):
            r = load_run(f"e4-{name}-{f}-highs-warm")
            cost_rows.append({"Instance": name, "Method": f"{NAME[f]} model, warm start", "Cost": f2(r["objective"]),
                              "Bound": f2(r["bound"]), "Gap": pct(r["gap"]),
                              "Status": "optimal" if r["optimal"] else f"best found, {r['time_limit']} s"})
            put(f"{key}_e4_{f}_cost", r["objective"], r["run"] + ".json")
            put(f"{key}_e4_{f}_bound", r["bound"], r["run"] + ".json")
            put(f"{key}_e4_{f}_gap", 100 * r["gap"], r["run"] + ".json", "{:.1f}")
            put(f"{key}_e4_{f}_nodes", f"{r['nodes']:,}", r["run"] + ".json")
        r = load_run(f"e5-{name}-fix-optimise")
        cost_rows.append({"Instance": name, "Method": "Fix-and-optimise", "Cost": f2(r["objective"]), "Bound": "",
                          "Gap": "", "Status": f"best found, {r['time']:.0f} s"})
        put(f"{key}_e5_cost", r["objective"], r["run"] + ".json")
        put(f"{key}_e5_start", r["start_objective"], r["run"] + ".json")
        put(f"{key}_e5_time", r["time"], r["run"] + ".json", "{:.0f}")
        put(f"{key}_e5_solves", r["subproblems"], r["run"] + ".json")
        put(f"{key}_e5_proven", r["subproblems_optimal"], r["run"] + ".json")
        put(f"{key}_e5_window", r["widest_window"], r["run"] + ".json")
        cost_rows.append({"Instance": name, "Method": "Best published", "Cost": f2(PUBLISHED[name]),
                          "Bound": "", "Gap": "", "Status": "best known"})
    table("t_e3", rows)
    table("t_benchmark", cost_rows)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    for ax, name in zip(axes, BENCHMARKS):
        sub = [r for r in cost_rows if r["Instance"] == name]
        labels = [r["Method"].replace(" (best of 5)", "").replace(", warm start", "\n(warm start)")
                  .replace("Start (", "Start\n(") for r in sub]
        vals = [float(r["Cost"]) for r in sub]
        y = np.arange(len(vals))[::-1]
        colours = ["#c3c2b7"] + [COLOUR["compact"]] * (len(vals) - 2) + ["#52514e"]
        ax.barh(y, vals, color=colours, height=0.6, edgecolor="white")
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.2f}", va="center", fontsize=8)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Cost per student")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "f_benchmark_cost")


def e6_outputs():
    exact = sorted((load_run(p.stem) for p in RUNS.glob("e6-exact-*.json")), key=lambda r: r["periods"])
    rows = [{"T": r["periods"], "Cost": f2(r["objective"]), "Status": status(r), "Time (s)": f"{r['time']:.1f}",
             "Nodes": f"{r['nodes']:,}"} for r in exact]
    table("t_e6_exact", rows)
    infeasible = [load_run(p.stem) for p in RUNS.glob("e6-feasible-*.json") if load_run(p.stem)["timetable"] is None]
    put("e6_n", exact[0]["exams"], exact[0]["run"] + ".json")
    put("e6_tmin", exact[0]["periods"], exact[0]["run"] + ".json")
    put("e6_tinfeasible", infeasible[0]["periods"], infeasible[0]["run"] + ".json")
    put("e6_tmax", exact[-1]["periods"], exact[-1]["run"] + ".json")
    put("e6_cost_tmin", exact[0]["objective"], exact[0]["run"] + ".json")
    put("e6_cost_tmax", exact[-1]["objective"], exact[-1]["run"] + ".json")
    put("e6_all_optimal", "yes" if all(r["optimal"] for r in exact) else "no", "e6-exact-*.json")

    heur = {}
    for T in range(18, 25):
        heur[T] = np.array([load_run(f"e6-hec-s-92-T{T}-seed{s}")["objective"] for s in range(3)])
    table("t_e6_heuristic", [{"T": T, "Mean": f2(v.mean()), "Best": f2(v.min()), "Worst": f2(v.max())}
                             for T, v in heur.items()])
    put("e6_heur_18", heur[18].min(), "e6-hec-s-92-T18-seed*.json")
    put("e6_heur_19", heur[19].min(), "e6-hec-s-92-T19-seed*.json")
    put("e6_heur_24", heur[24].min(), "e6-hec-s-92-T24-seed*.json")
    put("e6_heur_drop19", 100 * (1 - heur[19].min() / heur[18].min()), "e6-hec-s-92-T18/T19-seed*.json", "{:.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    T = [r["periods"] for r in exact]
    axes[0].plot(T, [r["objective"] for r in exact], marker="o", color=COLOUR["compact"], linewidth=2)
    axes[0].set_title(f"hec-s-92, n = {exact[0]['exams']}, proven optima", fontsize=9)
    axes[0].set_ylabel("Cost per student")
    Ts = list(heur)
    axes[1].fill_between(Ts, [heur[t].min() for t in Ts], [heur[t].max() for t in Ts], color="#2a78d6", alpha=0.25, linewidth=0, label="range of 3 seeds")
    axes[1].plot(Ts, [heur[t].mean() for t in Ts], marker="s", color=COLOUR["compact"], linewidth=2, label="mean of 3 seeds")
    axes[1].set_title("hec-s-92, full instance, annealing", fontsize=9)
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.set_xlabel("Number of periods T")
    save(fig, "f_periods")


def dept_outputs():
    ex, sa = load_run("dept-synthetic-strong-highs"), load_run("dept-synthetic-sa")
    table("t_dept", [
        {"Method": "Strengthened model, HiGHS", "Cost": f2(ex["objective"]), "Bound": f2(ex["bound"]),
         "Gap": pct(ex["gap"]), "Status": status(ex), "Time (s)": f"{ex['time']:.1f}"},
        {"Method": "Simulated annealing", "Cost": f2(sa["objective"]), "Bound": "", "Gap": "",
         "Status": "best found", "Time (s)": f"{sa['time_limit']:.1f}"}])
    inst = data.from_csv(data.ROOT / "data" / "synthetic_department.csv")
    st = inst.stats()
    for k in ("exams", "students", "enrolments", "conflicting_pairs"):
        put(f"dept_{k}", st[k], "data/synthetic_department.csv")
    put("dept_periods", ex["periods"], ex["run"] + ".json")
    put("dept_exact", ex["objective"], ex["run"] + ".json")
    put("dept_bound", ex["bound"], ex["run"] + ".json")
    put("dept_status", status(ex), ex["run"] + ".json")
    put("dept_sa", sa["objective"], sa["run"] + ".json")


def best_timetables():
    """Best timetable per instance, rescored by the evaluator."""
    for name in BENCHMARKS:
        inst = data.load(name)
        cands = [p.stem for p in RUNS.glob(f"e[345]-{name}-*.json")] + [p.stem for p in RUNS.glob(f"e6-{name}-T{inst.periods}-*.json")]
        runs = [load_run(c) for c in cands if load_run(c).get("timetable")]
        best = min(runs, key=lambda r: r["objective"])
        ok, cost, _ = evaluate(best["timetable"], inst.students)
        assert ok and abs(cost - best["objective"]) < 1e-9
        assert cost >= STA_BEST - 1e-9 or name != "sta-f-83", "below the best known value: check the scoring"
        size = np.bincount([e for s in inst.students for e in s], minlength=inst.n)
        df = pd.DataFrame({"exam_code": inst.exams, "period": np.array(best["timetable"]) + 1, "enrolment": size})
        df.to_csv(TIMETABLES / f"{name}.csv", index=False)
        by_period = df.groupby("period")
        table(f"t_timetable_{name}", [{"Period": p, "Exams": len(g), "Students sitting": f"{g['enrolment'].sum():,}",
                                       "Exam codes": " ".join(g["exam_code"])} for p, g in by_period])
        put(f"{name.split('-')[0]}_best", cost, best["run"] + ".json")
        put(f"{name.split('-')[0]}_best_run", best["run"], best["run"] + ".json")
        if name == "hec-s-92":
            fig, ax = plt.subplots(figsize=(6.3, 2.3))
            counts = by_period.size()
            ax.bar(counts.index, counts.values, color=COLOUR["compact"], width=0.7, edgecolor="white")
            ax.set_xticks(range(1, inst.periods + 1))
            ax.set_xlabel("Period")
            ax.set_ylabel("Exams")
            ax.grid(axis="x", visible=False)
            save(fig, "f_exams_per_period")
            put("hec_period_min", int(counts.min()), best["run"] + ".json")
            put("hec_period_max", int(counts.max()), best["run"] + ".json")
    ex = load_run("dept-synthetic-strong-highs")
    inst = data.from_csv(data.ROOT / "data" / "synthetic_department.csv")
    pd.DataFrame({"exam_code": inst.exams, "period": np.array(ex["timetable"]) + 1}).to_csv(
        TIMETABLES / "synthetic-department.csv", index=False)


def misc():
    r = load_run("pulp-status-hec-s-92")
    put("pulp_status", r["pulp_status"], r["run"] + ".json")
    put("pulp_cbc_status", r["cbc_status"], r["run"] + ".json")
    put("pulp_exams", r["exams"], r["run"] + ".json")
    put("pulp_limit", r["time_limit"], r["run"] + ".json")
    put("e6_time_T11", load_run("e6-exact-n10-T11")["time"], "e6-exact-n10-T11.json", "{:.1f}")
    for key, value in (("limit_e2", 120), ("limit_e4", 480), ("limit_e5", 480), ("sa_seeds", 5), ("sa_seconds", 60)):
        put(key, value, "src/experiments.py")
    T = 18
    put("lp_xval", 1 / T, "analytic, T = 18", "{:.4f}")
    put("hec_lp_time", load_run("lp-hec-s-92-strong")["time"], "lp-hec-s-92-strong.json", "{:.1f}")
    put("hec_compact_vars", f"{load_run('lp-hec-s-92-compact')['variables']:,}", "lp-hec-s-92-compact.json")
    put("hec_compact_cons", f"{load_run('lp-hec-s-92-compact')['constraints']:,}", "lp-hec-s-92-compact.json")
    put("hec_strong_cons", f"{load_run('lp-hec-s-92-strong')['constraints']:,}", "lp-hec-s-92-strong.json")
    pv, pc = pairwise_size(81, 1363, 18)
    put("hec_pairwise_vars", f"{pv:,}", "formula, checked against model.build_pairwise")
    put("hec_pairwise_cons", f"{pc:,}", "formula, checked against model.build_pairwise")
    put("hec_pairwise_ratio", pv / load_run("lp-hec-s-92-compact")["variables"], "formula / lp-hec-s-92-compact.json", "{:.0f}")
    for name in BENCHMARKS:
        put(f"{name.split('-')[0]}_cliques", len(model.cliques(data.load(name))), f"data/raw/{name}.*")


def results_files():
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    decisions = (data.ROOT / "DECISIONS.md").read_text() if (data.ROOT / "DECISIONS.md").exists() else ""
    lines = ["# Results", "", "Every number quoted in the report, with the run file it comes from "
             "(`outputs/runs/`). Generated by `src/figures.py`; do not edit by hand.", "",
             "| Key | Value | Source |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v} | {sources[k]} |" for k, v in results.items()]
    (data.ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n\n" + decisions)


if __name__ == "__main__":
    for folder in (TABLES, FIGS, TIMETABLES):
        folder.mkdir(parents=True, exist_ok=True)
    instances_table()
    degree_figure()
    e1_table()
    e2_figures(e2_tables())
    lp_table()
    benchmark_tables()
    e6_outputs()
    dept_outputs()
    best_timetables()
    misc()
    results_files()
    print(f"{len(results)} numbers, tables in {TABLES}, figures in {FIGS}")
