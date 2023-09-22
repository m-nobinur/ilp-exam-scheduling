"""Toronto benchmark parser, sub-instances and CSV loader."""
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PERIODS = {"hec-s-92": 18, "sta-f-83": 13}


@dataclass
class Instance:
    name: str
    exams: list
    students: list
    periods: int = None
    conflicts: dict = field(init=False)

    def __post_init__(self):
        self.conflicts = Counter(
            pair for s in self.students for pair in combinations(sorted(s), 2))

    @property
    def n(self):
        return len(self.exams)

    def matrix(self):
        C = np.zeros((self.n, self.n), dtype=np.int64)
        for (i, j), c in self.conflicts.items():
            C[i, j] = C[j, i] = c
        return C

    def stats(self):
        n, pairs = self.n, len(self.conflicts)
        return {"instance": self.name, "exams": n, "students": len(self.students),
                "enrolments": sum(len(s) for s in self.students),
                "conflicting_pairs": pairs,
                "density": round(2 * pairs / n**2, 3),
                "periods": self.periods}


def load(name):
    codes = [line.split()[0] for line in (RAW / f"{name}.crs").read_text().splitlines() if line.strip()]
    index = {c: i for i, c in enumerate(codes)}
    students = []
    for line in (RAW / f"{name}.stu").read_text().splitlines():
        if line.split():
            students.append(tuple(sorted({index[c] for c in line.split()})))
    return Instance(name, codes, students, PERIODS.get(name))


def sub_instance(inst, n):
    """The n exams with the largest enrolment, and the students who sit at least two of them."""
    size = Counter(e for s in inst.students for e in s)
    keep = sorted(sorted(range(inst.n), key=lambda e: (-size[e], e))[:n])
    new = {e: k for k, e in enumerate(keep)}
    students = [tuple(new[e] for e in s if e in new) for s in inst.students]
    students = [s for s in students if len(s) >= 2]
    return Instance(f"{inst.name}-n{n}", [inst.exams[e] for e in keep], students)


def from_csv(path, name=None, periods=None):
    """Load a department file with columns student_id, exam_code."""
    df = pd.read_csv(path, dtype=str).dropna().drop_duplicates()
    codes = sorted(df["exam_code"].unique())
    index = {c: i for i, c in enumerate(codes)}
    students = [tuple(sorted(index[c] for c in g)) for _, g in df.groupby("student_id")["exam_code"]]
    return Instance(name or Path(path).stem, codes, students, periods)


if __name__ == "__main__":
    expected = {"hec-s-92": (81, 2823, 10632, 1363, 0.415), "sta-f-83": (139, 611, 5751, 1381, 0.143)}
    for name, values in expected.items():
        st = load(name).stats()
        got = (st["exams"], st["students"], st["enrolments"], st["conflicting_pairs"], st["density"])
        assert got == values, (name, got)
        print(st)
