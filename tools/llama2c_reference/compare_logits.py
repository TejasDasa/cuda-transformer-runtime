#!/usr/bin/env python3
"""Compare two logits dumps ("pos token logit..." per line) and report
max abs diff, RMSE, relative-to-scale error and argmax agreement.

Usage: compare_logits.py ours.txt reference.txt [abs_threshold]
Exit 0 if every position is within the threshold (default 1e-2 absolute),
else 1.  Pure Python: no numpy required.
"""
import math
import sys


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            rows.append((int(parts[0]), int(parts[1]), [float(x) for x in parts[2:]]))
    return rows


def main():
    ours = load(sys.argv[1])
    ref = load(sys.argv[2])
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2
    if len(ours) != len(ref):
        print(f"position count differs: ours={len(ours)} ref={len(ref)}")
        return 1
    ok = True
    worst_abs = 0.0
    print(f"{'pos':>3} {'token':>6} {'n':>6} {'max_abs':>10} {'rmse':>10} {'max_rel_to_scale':>16} {'argmax':>8}")
    for (p1, t1, a), (p2, t2, b) in zip(ours, ref):
        if p1 != p2 or t1 != t2 or len(a) != len(b):
            print(f"row mismatch at pos {p1}: tokens {t1}/{t2}, lengths {len(a)}/{len(b)}")
            return 1
        scale = max(abs(x) for x in b) or 1.0
        diffs = [abs(x - y) for x, y in zip(a, b)]
        max_abs = max(diffs)
        rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        same_argmax = a.index(max(a)) == b.index(max(b))
        worst_abs = max(worst_abs, max_abs)
        ok &= max_abs <= threshold and same_argmax
        print(f"{p1:>3} {t1:>6} {len(a):>6} {max_abs:>10.3e} {rmse:>10.3e} {max_abs / scale:>16.3e} {'same' if same_argmax else 'DIFF':>8}")
    print(f"worst max_abs={worst_abs:.3e} threshold={threshold:g} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
