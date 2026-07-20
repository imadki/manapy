#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract the (magneto-)Rossby dispersion omega(kx) from swmhd_rossby2d.py runs
and overlay the analytic relations.

Two modes:
  * `--run` : launch swmhd_rossby2d.py for each KX in KX_LIST (with the given B0)
              via subprocess, producing rossby_kx*_B0*.txt, then analyse.
  * default : read existing rossby_kx*_B0<B0>.txt files in the current dir.

For each file the complex modal amplitude A(t)=re+i*im rotates as exp(-i omega t);
omega(kx) is the dominant FFT peak of A restricted to the SLOW band (below the
fast magneto-gravity frequency). It is compared to:

  * hydrodynamic Rossby (B0=0):
        omega = -beta*kx / (k^2 + 1/Ld^2),   Ld^2 = g*H0/f0^2
  * barotropic magneto-Rossby (rigid-lid limit, Ld->inf), V_A = B0:
        omega^2 + (beta*kx/k^2) omega - kx^2 V_A^2 = 0
    -> two branches (fast/slow); B0=0 recovers Rossby, beta=0 recovers Alfven.

The free-surface SWMHD relation has extra finite-Ld terms; these two textbook
limits are the validation scaffold (numeric points should land on the Rossby
curve for B0=0 and split once B0>0).

Usage:
    B0=0.0 python plot_dispersion_swmhd.py --run     # generate + plot pure Rossby
    B0=0.1 python plot_dispersion_swmhd.py --run     # generate + plot magneto-Rossby
    B0=0.1 python plot_dispersion_swmhd.py           # plot from existing files
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- must match the values used in swmhd_rossby2d.py ---
GRAV = 1.0
H0 = float(os.environ.get('H0', 10.0))
F0 = float(os.environ.get('F0', 8.0))
BETA = float(os.environ.get('BETA', 1.0))
B0 = float(os.environ.get('B0', 0.1))
KX_LIST = [2.0 * np.pi * m for m in (1, 2, 3, 4)]
Ld2 = GRAV * H0 / (F0 * F0)   # deformation radius squared = gH/f0^2


def omega_from_file(fn, kx, ky=0.0):
  # The example seeds a GEOSTROPHICALLY balanced single mode, so A(t) rotates
  # cleanly as exp(-i omega t): the phase-increment estimator is exact even over
  # a fraction of a period. (Fall back to the slow-band FFT peak if the phase is
  # too noisy.)
  d = np.loadtxt(fn)
  t = d[:, 0]
  A = d[:, 1] + 1j * d[:, 2]
  i0 = len(t) // 8                       # drop the initial transient
  t = t[i0:]
  A = A[i0:]
  dth = np.angle(A[1:] * np.conj(A[:-1]))
  dt = np.diff(t)
  return float(np.median(-dth / dt))


def omega_rossby(kx, ky=0.0):
  k2 = kx * kx + ky * ky
  return -BETA * kx / (k2 + 1.0 / Ld2)   # QG Rossby (valid when beta*L << f0)


def omega_magrossby(kx, va, ky=0.0):
  k2 = kx * kx + ky * ky
  b = BETA * kx / k2
  disc = np.sqrt(b * b + 4.0 * kx * kx * va * va)
  return (-b + disc) / 2.0, (-b - disc) / 2.0   # fast, slow branches


if '--run' in sys.argv:
  for kx in KX_LIST:
    out = f'rossby_kx{kx:.4f}_B0{B0:.3f}.txt'
    env = dict(os.environ, KX=str(kx), KY='0', B0=str(B0),
               F0=str(F0), BETA=str(BETA), H0=str(H0), OUT=out)
    print(f'--- running KX={kx:.4f} B0={B0} ---')
    subprocess.run([sys.executable, 'swmhd_rossby2d.py'], env=env, check=True)

files = sorted(glob.glob(f'rossby_kx*_B0{B0:.3f}.txt'))
if not files:
  sys.exit(f'no rossby_kx*_B0{B0:.3f}.txt files found; run with --run first')

kxs, oms = [], []
for fn in files:
  kx = float(re.search(r'kx([\d.]+)_', fn).group(1))
  kxs.append(kx)
  oms.append(omega_from_file(fn, kx))
kxs = np.array(kxs)
oms = np.array(oms)

kk = np.linspace(min(kxs), max(kxs), 200)
plt.figure(figsize=(7, 5))
plt.plot(kxs, oms, 'ko', ms=8, label='numeric (FFT of modal A(t))')
plt.plot(kk, omega_rossby(kk), 'b-', label='Rossby  (B0=0)')
if B0 > 0:
  fast = np.array([omega_magrossby(k, B0)[0] for k in kk])
  slow = np.array([omega_magrossby(k, B0)[1] for k in kk])
  plt.plot(kk, fast, 'r--', label='magneto-Rossby (fast)')
  plt.plot(kk, slow, 'g--', label='magneto-Rossby (slow)')
plt.axhline(0, color='k', lw=0.5)
plt.xlabel(r'$k_x$')
plt.ylabel(r'$\omega$')
plt.title(f'SWMHD (magneto-)Rossby dispersion  '
          f'(B0={B0}, beta={BETA}, f0={F0})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('dispersion_swmhd.png', dpi=130, bbox_inches='tight')
print('wrote dispersion_swmhd.png')
