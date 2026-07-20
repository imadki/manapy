#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two validate_fv.py dumps (typically cpu vs gpu, same scheme).

Usage:
  python3 compare_fv.py <prefixA> <prefixB>

Checks, from sharpest to softest:
  1. Assembled matrix  A_A vs A_B  (COO -> CSR sums duplicates, so triplet order
     is irrelevant). This is solver-independent and nails the triplet kernel.
  2. Assembled RHS.
  3. Final solution P.cell.

On a single rank cell_loctoglob is the identity, so the two COO index sets are
directly comparable.
"""
import sys

import numpy as np
from scipy.sparse import csr_matrix

if len(sys.argv) != 3:
  print(__doc__)
  sys.exit(1)
A, B = sys.argv[1], sys.argv[2]


def load_csr(prefix):
  row = np.load(f"{prefix}_row.npy")
  col = np.load(f"{prefix}_col.npy")
  data = np.load(f"{prefix}_data.npy")
  n = int(max(row.max(), col.max())) + 1
  M = csr_matrix((data, (row, col)), shape=(n, n))
  M.sum_duplicates()
  M.sort_indices()
  return M


def rel(a, b):
  na = np.linalg.norm(b)
  return np.linalg.norm(a - b) / (na if na else 1.0)


print(f"== compare {A}  vs  {B} ==")

# 1) matrix
Ma, Mb = load_csr(A), load_csr(B)
if Ma.shape != Mb.shape:
  print(f"  MATRIX: shape mismatch {Ma.shape} vs {Mb.shape}  -> FAIL")
  sys.exit(2)
dM = (Ma - Mb)
mrel = np.abs(dM.data).max() / (np.abs(Ma.data).max() if Ma.nnz else 1.0) if dM.nnz else 0.0
print(f"  matrix   nnz {Ma.nnz} vs {Mb.nnz} | max|dA|/max|A| = {mrel:.3e}"
      f"  -> {'OK' if mrel < 1e-10 else 'DIFF'}")

# 2) rhs
ra, rb = np.load(f"{A}_rhs.npy"), np.load(f"{B}_rhs.npy")
rrel = rel(ra, rb)
print(f"  rhs      |dr|/|r| = {rrel:.3e}  -> {'OK' if rrel < 1e-10 else 'DIFF'}")

# 3) solution
sa, sb = np.load(f"{A}_sol.npy"), np.load(f"{B}_sol.npy")
srel = rel(sa, sb)
print(f"  solution |dx|/|x| = {srel:.3e}  -> {'OK' if srel < 1e-6 else 'DIFF'}")

ok = (mrel < 1e-10) and (rrel < 1e-10) and (srel < 1e-6)
print("== VERDICT:", "MATCH" if ok else "MISMATCH", "==")
sys.exit(0 if ok else 3)
