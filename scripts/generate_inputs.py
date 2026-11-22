"""Generate the dense input matrices.

Same generator as Assignments 1-3: A_n.bin and B_n.bin contain n*n little-endian float64
values (row-major) drawn uniformly from [-1, 1) with numpy's default_rng(SEED + n),
A first and then B. For n = 800, 1600 and 3200 the files are therefore identical to
those of the earlier assignments; the SHA-256 of every file is stored in
results/raw/input_checksums.csv so this can be checked."""
import csv
import hashlib

import numpy as np

from common import INPUT_DIR, RAW_DIR, SEED, VALUE_RANGE, all_sizes, input_paths


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in all_sizes():
        rng = np.random.default_rng(SEED + n)
        A = rng.uniform(*VALUE_RANGE, size=(n, n))
        B = rng.uniform(*VALUE_RANGE, size=(n, n))
        for name, M, path in zip("AB", (A, B), input_paths(n)):
            M.astype("<f8").tofile(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"n": n, "matrix": name, "file": path.name, "sha256": digest})
            print(f"n={n}: {path.name} sha256={digest[:16]}...")
    with open(RAW_DIR / "input_checksums.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
