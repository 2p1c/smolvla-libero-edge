"""Compare two eval runs episode by episode.

With the same seed, suite and eval batch size, episode i of a task starts from the same initial state in
both runs, so the comparison is paired: count episodes that flipped fail->success and success->fail.
That is far more sensitive than comparing two averages.

    python scripts/compare_evals.py official_n1 official_n1_res256
    python scripts/compare_evals.py official_n1 official_n1_res256 --first 5
"""

import argparse
import json
from math import comb
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def successes(tag, first):
    info = json.loads((RESULTS / tag / "eval_info.json").read_text())
    return {t["task_id"]: [bool(s) for s in t["metrics"]["successes"][:first]] for t in info["per_task"]}


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant pairs (b: A only, c: B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--first", type=int, default=None, help="only the first N episodes of each task")
    args = parser.parse_args()

    a, b = successes(args.a, args.first), successes(args.b, args.first)
    print(f"| task | {args.a} | {args.b} | fail->success | success->fail |")
    print("|---|---|---|---|---|")
    tot_a = tot_b = n = gained = lost = 0
    for task in sorted(a):
        sa, sb = a[task], b[task]
        m = min(len(sa), len(sb))
        sa, sb = sa[:m], sb[:m]
        g = sum((not x) and y for x, y in zip(sa, sb))
        l = sum(x and (not y) for x, y in zip(sa, sb))
        print(f"| {task} | {100 * sum(sa) / m:.0f}% | {100 * sum(sb) / m:.0f}% | {g} | {l} |")
        tot_a, tot_b, n, gained, lost = tot_a + sum(sa), tot_b + sum(sb), n + m, gained + g, lost + l
    print(f"| all | {100 * tot_a / n:.1f}% | {100 * tot_b / n:.1f}% | {gained} | {lost} |")
    print(f"\n{n} paired episodes; exact McNemar p = {mcnemar_p(lost, gained):.3f}")


if __name__ == "__main__":
    main()
