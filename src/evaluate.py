"""Independent scorer. Uses raw student lists only, no code shared with the model."""
from itertools import combinations

WEIGHTS = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1}


def evaluate(timetable, students):
    """timetable: exam -> period.
    Returns (conflict_free, cost per student, total penalty)."""
    clashes = total = 0
    for exams in students:
        for a, b in combinations(exams, 2):
            gap = abs(timetable[a] - timetable[b])
            if gap == 0:
                clashes += 1
            total += WEIGHTS.get(gap, 0)
    return clashes == 0, total / len(students), total


if __name__ == "__main__":
    students = [("A", "B", "C"), ("B", "D"), ("C", "D", "E"), ("A", "E")]
    timetable = {"A": 1, "B": 2, "C": 4, "D": 5, "E": 3}
    assert evaluate(timetable, students) == (True, 20.0, 80)
    assert evaluate({**timetable, "E": 4}, students)[0] is False
    print("evaluator ok")
