#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-order WENO finite-volume reconstruction on unstructured meshes.

Implementation of Tsoutsanis, "Stencil selection algorithms for WENO schemes on
unstructured meshes", J. Comput. Phys. 475 (2023) 108840 (weno.pdf in this dir).

What is implemented (validated standalone):
  * k-exact least-squares reconstruction on the central (vertex-based) stencil:
    a polynomial p_i of order r built from the surrounding cell averages, with the
    zero-cell-average basis constraint (Eq 12) enforced through *exact* cell
    moments -> genuine k-exactness (degree <= r recovered exactly). The
    geometry-only pseudo-inverse A^+ (Eqs 8-16) is precomputed once per cell.
  * directional (sectoral) stencils drawn from the two-ring node-neighbour pool,
    one per direction, each with its own precomputed pseudo-inverse.
  * oscillation matrix OI (Eq 23) and smoothness indicator SI = a^T OI a (Eq 22),
    precomputed per cell; small for smooth data, large where data oscillate.
  * non-linear WENO weighting (Eqs 17-20): w_s = lam_s/(eps+SI_s)^power, with a
    large central linear weight. `weno_reconstruct` stays k-exact in smooth
    regions and essentially non-oscillatory at discontinuities (no Gibbs overshoot).

Everything that scales with the mesh is compiled with numba: the one-time build
(cell central moments, two-ring stencil selection, least-squares matrices,
SVD pseudo-inverses, oscillation matrices) runs in dedicated kernels, and so does
the per-step reconstruction/flux. Only the tiny mesh-independent "recipes" (the
binomial-shift and oscillation-matrix term lists, O(basis^2)) are built in Python.

Remaining: evaluate the WENO polynomial at edge Gauss points and wire it into the
Euler flux (replacing the MUSCL order-2 reconstruction); Shu-Osher validation.
"""
import numpy as np
from manapy.backends.compile_fun import compile


def _weno_kernel_2d(U: 'float64[:]', coeffs: 'float64[:,:]', st_idx: 'int32[:,:,:]',
                    st_cnt: 'int32[:,:]', pinv: 'float64[:,:,:,:]', OI: 'float64[:,:,:]',
                    lam: 'float64[:]', eps: 'float64', power: 'float64'):
    # Per-step WENO reconstruction hot loop (compiled). All geometry-dependent
    # arrays (stencil indices, padded pseudo-inverses, oscillation matrices,
    # linear weights) are precomputed once on the mesh and passed in; this only
    # does the data-dependent matvecs, smoothness quadratic forms and weighting.
    ncells = coeffs.shape[0]
    K = coeffs.shape[1]
    ns = lam.shape[0]
    a_s = np.zeros((ns, K))
    wbar = np.zeros(ns)
    for i in range(ncells):
        ui = U[i]
        for s in range(ns):
            cnt = st_cnt[i, s]
            for k in range(K):
                acc = 0.0
                for j in range(cnt):
                    acc += pinv[i, s, k, j] * (U[st_idx[i, s, j]] - ui)
                a_s[s, k] = acc
            si = 0.0
            for k in range(K):
                for q in range(K):
                    si += a_s[s, k] * OI[i, k, q] * a_s[s, q]
            wbar[s] = lam[s] / (eps + si) ** power
        wsum = 0.0
        for s in range(ns):
            wsum += wbar[s]
        for k in range(K):
            v = 0.0
            for s in range(ns):
                v += wbar[s] * a_s[s, k]
            coeffs[i, k] = v / wsum


_weno_kernel_2d_compiled = None


def _weno_build_2d(cm: 'float64[:,:]', cx: 'float64[:]', cy: 'float64[:]',
                   vol: 'float64[:]', h: 'float64[:]',
                   st_idx: 'int32[:,:,:]', st_cnt: 'int32[:,:]',
                   amom: 'int32[:,:]', apdx: 'int32[:,:]', apdy: 'int32[:,:]',
                   acoef: 'float64[:,:]', acnt: 'int32[:]', aord: 'int32[:]', mono_cmidx: 'int32[:]',
                   oik: 'int32[:]', oiq: 'int32[:]', oimom: 'int32[:]', oicoef: 'float64[:]', oiord: 'int32[:]',
                   pinv_p: 'float64[:,:,:,:]', OI_p: 'float64[:,:,:]'):
    # Per-cell WENO build (compiled): for each stencil, form the least-squares
    # geometry matrix A from the precomputed central moments (binomial shift by the
    # stencil-local offset), then its pseudo-inverse via SVD (robust to ill-
    # conditioning); also assemble the oscillation matrix OI. No O(ncells) temporaries.
    nc = pinv_p.shape[0]   # reconstruct local cells only (cm may be extended with halo rows)
    ns = st_cnt.shape[1]
    K = pinv_p.shape[2]
    max_m = pinv_p.shape[3]
    A = np.zeros((max_m, K))
    for i in range(nc):
        hi = h[i]; voli = vol[i]; xi = cx[i]; yi = cy[i]
        for s in range(ns):
            cnt = st_cnt[i, s]
            for j in range(cnt):
                m = st_idx[i, s, j]
                dx = cx[m] - xi; dy = cy[m] - yi
                volm = vol[m]
                for k in range(K):
                    integ = 0.0
                    for tt in range(acnt[k]):
                        integ += acoef[k, tt] * dx ** apdx[k, tt] * dy ** apdy[k, tt] * cm[m, amom[k, tt]]
                    avg_m = integ / (volm * hi ** aord[k])
                    m0k = cm[i, mono_cmidx[k]] / (voli * hi ** aord[k])
                    A[j, k] = avg_m - m0k
            # pseudo-inverse of A[:cnt, :K] via SVD: pinv = V diag(1/s) U^T. The
            # near-null truncation (1e-10*s0) keeps full k-exactness on well-
            # conditioned stencils and bounds the ~degenerate ones (sliver cells, or
            # partition-boundary cells whose halo pool is only partial under MPI).
            U, sv, Vt = np.linalg.svd(A[:cnt, :])
            tol = 1e-10 * sv[0]
            for k in range(K):
                for j in range(cnt):
                    acc = 0.0
                    for l in range(K):
                        if sv[l] > tol:
                            acc += Vt[l, k] * (1.0 / sv[l]) * U[j, l]
                    pinv_p[i, s, k, j] = acc
        # oscillation matrix OI[i] from cell-i central moments
        for t in range(oik.shape[0]):
            OI_p[i, oik[t], oiq[t]] += oicoef[t] * cm[i, oimom[t]] / (voli * hi ** oiord[t])


_weno_build_2d_compiled = None


def _weno_cm_2d(nodeid: 'int32[:,:]', vx: 'float64[:]', vy: 'float64[:]',
                cx: 'float64[:]', cy: 'float64[:]', cm: 'float64[:,:]'):
    # Central moments (order <= 2) of each cell about its own centre, by fan
    # triangulation. cm columns are [00,10,01,20,11,02].
    nc = nodeid.shape[0]
    last = nodeid.shape[1] - 1
    for i in range(nc):
        nv = nodeid[i, last]
        cmx = cx[i]; cmy = cy[i]
        x0 = vx[nodeid[i, 0]] - cmx; y0 = vy[nodeid[i, 0]] - cmy
        for k in range(1, nv - 1):
            x1 = vx[nodeid[i, k]] - cmx; y1 = vy[nodeid[i, k]] - cmy
            x2 = vx[nodeid[i, k + 1]] - cmx; y2 = vy[nodeid[i, k + 1]] - cmy
            area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            cm[i, 0] += area
            cm[i, 1] += area * (x0 + x1 + x2) / 3.0
            cm[i, 2] += area * (y0 + y1 + y2) / 3.0
            cm[i, 3] += area / 6.0 * (x0 * x0 + x1 * x1 + x2 * x2 + x0 * x1 + x1 * x2 + x2 * x0)
            cm[i, 4] += area / 12.0 * (2 * (x0 * y0 + x1 * y1 + x2 * y2)
                                       + x0 * y1 + x0 * y2 + x1 * y0 + x1 * y2 + x2 * y0 + x2 * y1)
            cm[i, 5] += area / 6.0 * (y0 * y0 + y1 * y1 + y2 * y2 + y0 * y1 + y1 * y2 + y2 * y0)


def _weno_select_2d(cellnid: 'int32[:,:]', halonid: 'int32[:,:]', nb: 'int32',
                    cx: 'float64[:]', cy: 'float64[:]',
                    dirx: 'float64[:]', diry: 'float64[:]', m_central: 'int32', m_dir: 'int32',
                    st_idx: 'int32[:,:,:]', st_cnt: 'int32[:,:]'):
    # Build the central + directional stencils from the two-ring node-neighbour pool.
    # Halo (other-rank) node-neighbours from `halonid` are added to the pool, encoded
    # as nb + halo_id so the build/reconstruct kernels read them from the extended
    # geometry / field arrays. In serial (all halonid counts 0) this is a no-op and
    # reproduces the local-only stencils exactly. cx, cy are the EXTENDED positions.
    nc = cellnid.shape[0]
    last = cellnid.shape[1] - 1
    lasth = halonid.shape[1] - 1
    ndir = dirx.shape[0]
    poolmax = (last + 1) * (last + 1) + (last + 2) * (lasth + 1) + 8
    pool = np.empty(poolmax, dtype=np.int32)
    dist = np.empty(poolmax)
    score = np.empty(poolmax)
    for i in range(nc):
        npc = 0
        c1 = cellnid[i, last]
        for a in range(c1):
            m = cellnid[i, a]
            for b in range(-1, cellnid[m, last]):
                cand = m if b == -1 else cellnid[m, b]
                if cand == i:
                    continue
                dup = False
                for q in range(npc):
                    if pool[q] == cand:
                        dup = True; break
                if not dup and npc < poolmax:
                    pool[npc] = cand; npc += 1
        # halo node-neighbours of cell i and of its local one-ring (encoded nb+h)
        for a in range(-1, c1):
            m = i if a == -1 else cellnid[i, a]
            for b in range(halonid[m, lasth]):
                cand = nb + halonid[m, b]
                dup = False
                for q in range(npc):
                    if pool[q] == cand:
                        dup = True; break
                if not dup and npc < poolmax:
                    pool[npc] = cand; npc += 1
        dmax = 1e-30
        for p in range(npc):
            dx = cx[pool[p]] - cx[i]; dy = cy[pool[p]] - cy[i]
            dist[p] = (dx * dx + dy * dy) ** 0.5
            if dist[p] > dmax:
                dmax = dist[p]
        # central stencil: the m_central closest cells
        order = np.argsort(dist[:npc])
        nce = m_central if m_central < npc else npc
        st_cnt[i, 0] = nce
        for j in range(nce):
            st_idx[i, 0, j] = pool[order[j]]
        # directional stencils: most aligned with each direction
        for k in range(ndir):
            ex = dirx[k]; ey = diry[k]
            for p in range(npc):
                dx = cx[pool[p]] - cx[i]; dy = cy[pool[p]] - cy[i]
                al = (dx * ex + dy * ey) / (dist[p] if dist[p] > 1e-30 else 1e-30)
                score[p] = -(al - 0.1 * dist[p] / dmax)     # ascending sort of -score
            od = np.argsort(score[:npc])
            nde = m_dir if m_dir < npc else npc
            st_cnt[i, 1 + k] = nde
            for j in range(nde):
                st_idx[i, 1 + k, j] = pool[od[j]]


_weno_cm_2d_compiled = None
_weno_select_2d_compiled = None


def _weno_cm_3d(cellfid: 'int32[:,:]', facenid: 'int32[:,:]',
                vx: 'float64[:]', vy: 'float64[:]', vz: 'float64[:]',
                cx: 'float64[:]', cy: 'float64[:]', cz: 'float64[:]', cm: 'float64[:,:]'):
    # Central moments (degree <= 2) of each polyhedral cell about its own centre.
    # The convex cell is star-shaped from its centroid, so it tiles exactly into
    # tetrahedra {centroid, t0, t1, t2} over a fan-triangulation of each face. Each
    # tet moment uses the simplex formula (apex at the origin = centroid-shifted).
    # cm columns are [000,100,010,001,200,110,101,020,011,002].
    nc = cellfid.shape[0]
    lastc = cellfid.shape[1] - 1
    lastf = facenid.shape[1] - 1
    for i in range(nc):
        nf = cellfid[i, lastc]
        mx = cx[i]; my = cy[i]; mz = cz[i]
        for jf in range(nf):
            f = cellfid[i, jf]
            nv = facenid[f, lastf]
            n0 = facenid[f, 0]
            ax0 = vx[n0] - mx; ay0 = vy[n0] - my; az0 = vz[n0] - mz
            for k in range(1, nv - 1):
                nb = facenid[f, k]; ncid = facenid[f, k + 1]
                bx = vx[nb] - mx; by = vy[nb] - my; bz = vz[nb] - mz
                cx3 = vx[ncid] - mx; cy3 = vy[ncid] - my; cz3 = vz[ncid] - mz
                # tet {0, a, b, c} with a=(ax0,..), b, c ; V = |a . (b x c)| / 6
                det = (ax0 * (by * cz3 - bz * cy3)
                       - ay0 * (bx * cz3 - bz * cx3)
                       + az0 * (bx * cy3 - by * cx3))
                V = abs(det) / 6.0
                sx = ax0 + bx + cx3; sy = ay0 + by + cy3; sz = az0 + bz + cz3
                cm[i, 0] += V
                cm[i, 1] += V * 0.25 * sx
                cm[i, 2] += V * 0.25 * sy
                cm[i, 3] += V * 0.25 * sz
                cm[i, 4] += V / 20.0 * (ax0 * ax0 + bx * bx + cx3 * cx3 + sx * sx)
                cm[i, 5] += V / 20.0 * (ax0 * ay0 + bx * by + cx3 * cy3 + sx * sy)
                cm[i, 6] += V / 20.0 * (ax0 * az0 + bx * bz + cx3 * cz3 + sx * sz)
                cm[i, 7] += V / 20.0 * (ay0 * ay0 + by * by + cy3 * cy3 + sy * sy)
                cm[i, 8] += V / 20.0 * (ay0 * az0 + by * bz + cy3 * cz3 + sy * sz)
                cm[i, 9] += V / 20.0 * (az0 * az0 + bz * bz + cz3 * cz3 + sz * sz)


def _weno_select_3d(cellnid: 'int32[:,:]', halonid: 'int32[:,:]', nb: 'int32',
                    cx: 'float64[:]', cy: 'float64[:]', cz: 'float64[:]',
                    dirx: 'float64[:]', diry: 'float64[:]', dirz: 'float64[:]',
                    m_central: 'int32', m_dir: 'int32', st_idx: 'int32[:,:,:]', st_cnt: 'int32[:,:]'):
    # Central + directional stencils from the two-ring node-neighbour pool (3D).
    # Halo node-neighbours from `halonid` are added, encoded nb+halo_id (a no-op in
    # serial). cx, cy, cz are the EXTENDED positions.
    nc = cellnid.shape[0]
    last = cellnid.shape[1] - 1
    lasth = halonid.shape[1] - 1
    ndir = dirx.shape[0]
    poolmax = (last + 1) * (last + 1) + (last + 2) * (lasth + 1) + 8
    pool = np.empty(poolmax, dtype=np.int32)
    dist = np.empty(poolmax)
    score = np.empty(poolmax)
    for i in range(nc):
        npc = 0
        c1 = cellnid[i, last]
        for a in range(c1):
            m = cellnid[i, a]
            for b in range(-1, cellnid[m, last]):
                cand = m if b == -1 else cellnid[m, b]
                if cand == i:
                    continue
                dup = False
                for q in range(npc):
                    if pool[q] == cand:
                        dup = True; break
                if not dup and npc < poolmax:
                    pool[npc] = cand; npc += 1
        for a in range(-1, c1):
            m = i if a == -1 else cellnid[i, a]
            for b in range(halonid[m, lasth]):
                cand = nb + halonid[m, b]
                dup = False
                for q in range(npc):
                    if pool[q] == cand:
                        dup = True; break
                if not dup and npc < poolmax:
                    pool[npc] = cand; npc += 1
        dmax = 1e-30
        for p in range(npc):
            dx = cx[pool[p]] - cx[i]; dy = cy[pool[p]] - cy[i]; dz = cz[pool[p]] - cz[i]
            dist[p] = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist[p] > dmax:
                dmax = dist[p]
        order = np.argsort(dist[:npc])
        nce = m_central if m_central < npc else npc
        st_cnt[i, 0] = nce
        for j in range(nce):
            st_idx[i, 0, j] = pool[order[j]]
        for k in range(ndir):
            ex = dirx[k]; ey = diry[k]; ez = dirz[k]
            for p in range(npc):
                dx = cx[pool[p]] - cx[i]; dy = cy[pool[p]] - cy[i]; dz = cz[pool[p]] - cz[i]
                al = (dx * ex + dy * ey + dz * ez) / (dist[p] if dist[p] > 1e-30 else 1e-30)
                score[p] = -(al - 0.1 * dist[p] / dmax)
            od = np.argsort(score[:npc])
            nde = m_dir if m_dir < npc else npc
            st_cnt[i, 1 + k] = nde
            for j in range(nde):
                st_idx[i, 1 + k, j] = pool[od[j]]


def _weno_build_3d(cm: 'float64[:,:]', cx: 'float64[:]', cy: 'float64[:]', cz: 'float64[:]',
                   vol: 'float64[:]', h: 'float64[:]',
                   st_idx: 'int32[:,:,:]', st_cnt: 'int32[:,:]',
                   amom: 'int32[:,:]', apdx: 'int32[:,:]', apdy: 'int32[:,:]', apdz: 'int32[:,:]',
                   acoef: 'float64[:,:]', acnt: 'int32[:]', aord: 'int32[:]', mono_cmidx: 'int32[:]',
                   oik: 'int32[:]', oiq: 'int32[:]', oimom: 'int32[:]', oicoef: 'float64[:]', oiord: 'int32[:]',
                   pinv_p: 'float64[:,:,:,:]', OI_p: 'float64[:,:,:]'):
    # 3D per-cell WENO build (compiled): least-squares geometry matrix A from the
    # precomputed central moments (binomial shift by the stencil offset), SVD
    # pseudo-inverse, and the oscillation matrix OI. Mirrors _weno_build_2d with the
    # extra z factor.
    nc = pinv_p.shape[0]   # reconstruct local cells only (cm may be extended with halo rows)
    ns = st_cnt.shape[1]
    K = pinv_p.shape[2]
    max_m = pinv_p.shape[3]
    A = np.zeros((max_m, K))
    for i in range(nc):
        hi = h[i]; voli = vol[i]; xi = cx[i]; yi = cy[i]; zi = cz[i]
        for s in range(ns):
            cnt = st_cnt[i, s]
            for j in range(cnt):
                m = st_idx[i, s, j]
                dx = cx[m] - xi; dy = cy[m] - yi; dz = cz[m] - zi
                volm = vol[m]
                for k in range(K):
                    integ = 0.0
                    for tt in range(acnt[k]):
                        integ += acoef[k, tt] * dx ** apdx[k, tt] * dy ** apdy[k, tt] * dz ** apdz[k, tt] * cm[m, amom[k, tt]]
                    avg_m = integ / (volm * hi ** aord[k])
                    m0k = cm[i, mono_cmidx[k]] / (voli * hi ** aord[k])
                    A[j, k] = avg_m - m0k
            U, sv, Vt = np.linalg.svd(A[:cnt, :])
            # Truncate near-null singular directions. On a well-conditioned stencil
            # every singular value is O(sv[0]) so nothing is cut and the
            # reconstruction stays k-exact; on the ~3% of near-degenerate 3D stencils
            # (sliver/boundary cells) the tiny singular values are dropped, giving a
            # bounded minimum-norm (lower-order) fit instead of a blow-up.
            tol = 1e-10 * sv[0]
            for k in range(K):
                for j in range(cnt):
                    acc = 0.0
                    for l in range(K):
                        if sv[l] > tol:
                            acc += Vt[l, k] * (1.0 / sv[l]) * U[j, l]
                    pinv_p[i, s, k, j] = acc
        for t in range(oik.shape[0]):
            OI_p[i, oik[t], oiq[t]] += oicoef[t] * cm[i, oimom[t]] / (voli * hi ** oiord[t])


_weno_cm_3d_compiled = None
_weno_select_3d_compiled = None
_weno_build_3d_compiled = None


def _weno_advection_2d(rez: 'float64[:]', u_c: 'float64[:]', u_g: 'float64[:]',
                       coeffs: 'float64[:,:]', ea: 'int32[:]', eb: 'int32[:]', M0: 'float64[:,:]',
                       cx: 'float64[:]', cy: 'float64[:]', h: 'float64[:]',
                       fcx: 'float64[:]', fcy: 'float64[:]',
                       cellid: 'int32[:,:]', normal: 'float64[:,:]', mesure: 'float64[:]',
                       name: 'uint32[:]', ax: 'float64', ay: 'float64'):
    # Linear-advection residual with a WENO-reconstructed upwind face value:
    # evaluate the WENO polynomial of the upwind cell at the face centre. The
    # high-order, non-oscillatory reconstruction is what gives WENO its quality.
    K = coeffs.shape[1]
    rez[:] = np.zeros(len(rez))
    nbface = len(cellid)
    for f in range(nbface):
        mes = mesure[f]
        nx = normal[f][0] / mes
        ny = normal[f][1] / mes
        un = ax * nx + ay * ny
        il = cellid[f][0]
        inner = name[f] == 0
        # pick the upwind cell and evaluate its reconstruction at the face centre
        if un >= 0.0:
            ic = il
        elif inner:
            ic = cellid[f][1]
        else:
            ic = -1                                 # boundary, use ghost below
        if ic >= 0:
            val = u_c[ic]
            xi = cx[ic]; yi = cy[ic]; hi = h[ic]
            for k in range(K):
                val += coeffs[ic, k] * (((fcx[f] - xi) / hi) ** ea[k]
                                        * ((fcy[f] - yi) / hi) ** eb[k] - M0[ic, k])
        else:
            val = u_g[f]
        flux = un * val * mes
        rez[il] -= flux
        if inner:
            rez[cellid[f][1]] += flux


_weno_advection_2d_compiled = None


# scaled monomial exponents for order r (2D), excluding the constant term
_EXPONENTS = {
    1: [(1, 0), (0, 1)],
    2: [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)],
    3: [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2),
        (3, 0), (2, 1), (1, 2), (0, 3)],
}

# 3D monomial exponents for order r (excluding the constant term)
_EXPONENTS_3D = {
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
}


# central-moment column layout -- MUST match the hard-coded column order the
# _weno_cm_{2,3}d kernels write into.
_CM_MOMS = {
    2: [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)],
    3: [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
}


def _monomials_up_to(order, dim):
  """Central-moment column layout for a given dimension (matches the cm kernels)."""
  return _CM_MOMS[dim]


def _tri_moment(v0, v1, v2, a, b):
  """Exact integral of x^a y^b over a triangle (a+b <= 2), vertices v0,v1,v2."""
  x0, y0 = v0; x1, y1 = v1; x2, y2 = v2
  # absolute area: the per-vertex factors below are orientation-independent, only
  # the area magnitude enters the true geometric integral.
  area = 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
  if a == 0 and b == 0:
    return area
  if a == 1 and b == 0:
    return area * (x0 + x1 + x2) / 3.0
  if a == 0 and b == 1:
    return area * (y0 + y1 + y2) / 3.0
  if a == 2 and b == 0:
    return area / 6.0 * (x0 * x0 + x1 * x1 + x2 * x2 + x0 * x1 + x1 * x2 + x2 * x0)
  if a == 0 and b == 2:
    return area / 6.0 * (y0 * y0 + y1 * y1 + y2 * y2 + y0 * y1 + y1 * y2 + y2 * y0)
  if a == 1 and b == 1:
    return area / 12.0 * (2 * (x0 * y0 + x1 * y1 + x2 * y2)
                          + x0 * y1 + x0 * y2 + x1 * y0 + x1 * y2 + x2 * y0 + x2 * y1)
  raise ValueError("moment order > 2 not implemented")


class WenoReconstruction:

  def __init__(self, domain, order=2, ndir=None, lambda_central=1000.0, eps=1e-6, power=4.0):
    self.domain = domain
    self.dim = int(getattr(domain, "dim", 2))
    self.order = int(order)
    if self.dim == 2:
      if self.order not in _EXPONENTS:
        raise ValueError("order must be 1, 2 or 3 in 2D")
      self.exps = _EXPONENTS[self.order]
    else:
      if self.order not in _EXPONENTS_3D:
        raise ValueError("order must be 1 or 2 in 3D")
      self.exps = _EXPONENTS_3D[self.order]
    self.K = len(self.exps)

    if ndir is None:
      ndir = 4 if self.dim == 2 else 6                # angular (2D) / axis (3D) stencils
    self.ndir = int(ndir)
    self.lambda_central = float(lambda_central)
    self.eps = float(eps)
    self.power = float(power)

    cells = domain.cells
    self.nbcells = domain.nbcells
    self.center = np.asarray(cells.center)[:, :self.dim]
    self.vol = np.asarray(cells.volume)
    self.h = self.vol ** (1.0 / self.dim)           # per-cell length scale
    nc = self.nbcells
    ns = 1 + self.ndir

    # central-moment column layout and monomial->column map (dimension-generic)
    self._cm_moms = _monomials_up_to(self.order, self.dim)
    self._cm_idx = {m: k for k, m in enumerate(self._cm_moms)}
    ncm = len(self._cm_moms)

    m_central = int(np.ceil(1.8 * self.K))
    m_dir = int(np.ceil(1.5 * self.K))
    max_m = max(m_central, m_dir)
    self._st_idx = np.zeros((nc, ns, max_m), dtype=np.int32)
    self._st_cnt = np.zeros((nc, ns), dtype=np.int32)
    self._cm = np.zeros((nc, ncm))
    cellnid = np.ascontiguousarray(np.asarray(cells.cellnid), dtype=np.int32)
    verts = np.asarray(domain.nodes.vertex)
    self._pinv_p = np.zeros((nc, ns, self.K, max_m))
    self._OI_p = np.zeros((nc, self.K, self.K))

    amom, apd, acoef, acnt, aord, mono_cmidx = self._moment_recipe()
    oik, oiq, oimom, oicoef, oiord = self._oi_recipe()

    # local central moments + the directional unit vectors (dim-specific)
    if self.dim == 2:
      dirx, diry, dirz = self._cm_and_dirs_2d(cells, verts)
    else:
      dirx, diry, dirz = self._cm_and_dirs_3d(cells, verts)

    # --- extended [local | halo] geometry: halo (other-rank) cells are appended so
    # a stencil that crosses a partition boundary is complete under MPI. Their
    # positions/volumes come from halos.centvol and their central moments are
    # exchanged once here (mesh-only, so a single exchange at build time). ---
    self.nb = nb = nc
    self.nh = nh = int(getattr(domain, "nbhalos", 0))
    halonid = np.asarray(getattr(cells, "halonid", np.zeros((nc, 1))), dtype=np.int32)
    if halonid.ndim != 2 or halonid.shape[1] < 1:
      halonid = np.zeros((nc, 1), dtype=np.int32)
    halonid = np.ascontiguousarray(halonid)
    cxl = np.ascontiguousarray(self.center[:, 0]); cyl = np.ascontiguousarray(self.center[:, 1])
    czl = np.ascontiguousarray(self.center[:, 2]) if self.dim == 3 else np.zeros(nc)
    if nh > 0:
      centvol = np.asarray(domain.halos.centvol)          # [x, y, z, vol] per halo cell
      hvol = np.ascontiguousarray(centvol[:, 3])
      hh = hvol ** (1.0 / self.dim)
      halo_cm = np.zeros((nh, ncm))
      buf = np.zeros(nh)
      for col in range(ncm):
        domain.halo_comm.exchange(np.ascontiguousarray(self._cm[:, col]), recv_buffer=buf)
        halo_cm[:, col] = buf
      cx_e = np.concatenate([cxl, np.ascontiguousarray(centvol[:, 0])])
      cy_e = np.concatenate([cyl, np.ascontiguousarray(centvol[:, 1])])
      cz_e = np.concatenate([czl, np.ascontiguousarray(centvol[:, 2])])
      vol_e = np.concatenate([self.vol, hvol])
      h_e = np.concatenate([self.h, hh])
      cm_e = np.vstack([self._cm, halo_cm])
    else:
      cx_e, cy_e, cz_e = cxl, cyl, czl
      vol_e, h_e, cm_e = self.vol, self.h, self._cm
    self._uhalo = np.zeros(nh)

    # stencil selection (halo-aware) + per-cell build, over the extended geometry
    if self.dim == 2:
      self._weno_select(cellnid, halonid, np.int32(nb), cx_e, cy_e, dirx, diry,
                        np.int32(m_central), np.int32(m_dir), self._st_idx, self._st_cnt)
      self._weno_build(cm_e, cx_e, cy_e, vol_e, h_e, self._st_idx, self._st_cnt,
                       amom, apd[0], apd[1], acoef, acnt, aord, mono_cmidx,
                       oik, oiq, oimom, oicoef, oiord, self._pinv_p, self._OI_p)
    else:
      self._weno_select(cellnid, halonid, np.int32(nb), cx_e, cy_e, cz_e, dirx, diry, dirz,
                        np.int32(m_central), np.int32(m_dir), self._st_idx, self._st_cnt)
      self._weno_build(cm_e, cx_e, cy_e, cz_e, vol_e, h_e, self._st_idx, self._st_cnt,
                       amom, apd[0], apd[1], apd[2], acoef, acnt, aord, mono_cmidx,
                       oik, oiq, oimom, oicoef, oiord, self._pinv_p, self._OI_p)

    # M0 (subtracted mean of each basis monomial) for local + halo cells; the flux
    # kernels evaluate a halo neighbour's polynomial with the extended M0.
    nce = cm_e.shape[0]
    self._M0_ext = np.empty((nce, self.K))
    for k, e in enumerate(self.exps):
      self._M0_ext[:, k] = cm_e[:, self._cm_idx[e]] / (vol_e * h_e ** sum(e))
    self._M0_p = np.ascontiguousarray(self._M0_ext[:nc])
    self._cx_ext = cx_e; self._cy_ext = cy_e; self._cz_ext = cz_e; self._h_ext = h_e
    # halo-only geometry slices (for evaluating a halo neighbour's WENO polynomial
    # at a partition face, giving a full-order flux there under MPI)
    self._cx_h = np.ascontiguousarray(cx_e[nc:]); self._cy_h = np.ascontiguousarray(cy_e[nc:])
    self._cz_h = np.ascontiguousarray(cz_e[nc:]); self._h_h = np.ascontiguousarray(h_e[nc:])
    self._M0_h = np.ascontiguousarray(self._M0_ext[nc:])

    self._lam_arr = np.array([self.lambda_central] + [1.0] * self.ndir)
    global _weno_kernel_2d_compiled                  # hot loop is dimension-agnostic
    if _weno_kernel_2d_compiled is None:
      _weno_kernel_2d_compiled = compile(_weno_kernel_2d)
    self._kernel = _weno_kernel_2d_compiled

    # packed basis exponents / face geometry (used by evaluate and the flux kernels)
    self._eexp = np.array([list(e) for e in self.exps], dtype=np.int32)  # (K, dim)
    fc = np.asarray(self.domain.faces.center)
    self._ea = np.ascontiguousarray(self._eexp[:, 0]); self._eb = np.ascontiguousarray(self._eexp[:, 1])
    self._fcx = np.ascontiguousarray(fc[:, 0]); self._fcy = np.ascontiguousarray(fc[:, 1])
    self._cx = np.ascontiguousarray(self.center[:, 0]); self._cy = np.ascontiguousarray(self.center[:, 1])
    if self.dim == 3:
      self._ec = np.ascontiguousarray(self._eexp[:, 2])
      self._fcz = np.ascontiguousarray(fc[:, 2])
      self._cz = np.ascontiguousarray(self.center[:, 2])
    if self.dim == 2:
      global _weno_advection_2d_compiled
      if _weno_advection_2d_compiled is None:
        _weno_advection_2d_compiled = compile(_weno_advection_2d)
      self._adv_kernel = _weno_advection_2d_compiled

  def _cm_and_dirs_2d(self, cells, verts):
    """Compute the local cells' central moments (self._cm) and bind the compiled
    2D select/build kernels; return the ndir angular unit-direction vectors."""
    cx = np.ascontiguousarray(self.center[:, 0]); cy = np.ascontiguousarray(self.center[:, 1])
    nodeid = np.ascontiguousarray(np.asarray(cells.nodeid), dtype=np.int32)
    vx = np.ascontiguousarray(verts[:, 0]); vy = np.ascontiguousarray(verts[:, 1])
    global _weno_cm_2d_compiled, _weno_select_2d_compiled, _weno_build_2d_compiled
    if _weno_cm_2d_compiled is None:
      _weno_cm_2d_compiled = compile(_weno_cm_2d)
      _weno_select_2d_compiled = compile(_weno_select_2d)
      _weno_build_2d_compiled = compile(_weno_build_2d)
    _weno_cm_2d_compiled(nodeid, vx, vy, cx, cy, self._cm)
    self._weno_select = _weno_select_2d_compiled
    self._weno_build = _weno_build_2d_compiled
    ang = 2 * np.pi * np.arange(self.ndir) / self.ndir
    return np.ascontiguousarray(np.cos(ang)), np.ascontiguousarray(np.sin(ang)), None

  def _cm_and_dirs_3d(self, cells, verts):
    """Compute the local cells' central moments (self._cm) and bind the compiled
    3D select/build kernels; return the ndir axis-aligned unit-direction vectors."""
    cx = np.ascontiguousarray(self.center[:, 0]); cy = np.ascontiguousarray(self.center[:, 1])
    cz = np.ascontiguousarray(self.center[:, 2])
    cellfid = np.ascontiguousarray(np.asarray(cells.faceid), dtype=np.int32)
    facenid = np.ascontiguousarray(np.asarray(self.domain.faces.nodeid), dtype=np.int32)
    vx = np.ascontiguousarray(verts[:, 0]); vy = np.ascontiguousarray(verts[:, 1]); vz = np.ascontiguousarray(verts[:, 2])
    global _weno_cm_3d_compiled, _weno_select_3d_compiled, _weno_build_3d_compiled
    if _weno_cm_3d_compiled is None:
      _weno_cm_3d_compiled = compile(_weno_cm_3d)
      _weno_select_3d_compiled = compile(_weno_select_3d)
      _weno_build_3d_compiled = compile(_weno_build_3d)
    _weno_cm_3d_compiled(cellfid, facenid, vx, vy, vz, cx, cy, cz, self._cm)
    self._weno_select = _weno_select_3d_compiled
    self._weno_build = _weno_build_3d_compiled
    base = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0], [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
    if self.ndir <= 6:
      dirs = base[:self.ndir]
    else:
      extra = np.array([[s, t, u] for s in (1., -1) for t in (1., -1) for u in (1., -1)]) / np.sqrt(3)
      dirs = np.vstack([base, extra])[:self.ndir]
    return (np.ascontiguousarray(dirs[:, 0]), np.ascontiguousarray(dirs[:, 1]),
            np.ascontiguousarray(dirs[:, 2]))

  def _moment_recipe(self):
    """Per-monomial binomial-shift recipe (dimension-generic): the integral of a
    basis monomial about a shifted target is a sum of terms
    coef * prod_d d[d]^pd[d] * (central moment cm[mom]). Returns amom, a list of
    per-dimension power arrays apd, acoef, acnt, aord, mono_cmidx."""
    from math import comb
    from itertools import product
    d = self.dim
    K = self.K
    terms = []
    for e in self.exps:
      tk = []
      for p in product(*[range(ei + 1) for ei in e]):
        coef = 1.0
        for ei, pi in zip(e, p):
          coef *= comb(ei, pi)
        pdiff = tuple(ei - pi for ei, pi in zip(e, p))
        tk.append((self._cm_idx[p], pdiff, float(coef)))
      terms.append(tk)
    mt = max(len(tk) for tk in terms)
    amom = np.zeros((K, mt), np.int32); acoef = np.zeros((K, mt))
    apd = [np.zeros((K, mt), np.int32) for _ in range(d)]
    acnt = np.zeros(K, np.int32)
    aord = np.array([sum(e) for e in self.exps], np.int32)
    mono_cmidx = np.array([self._cm_idx[e] for e in self.exps], np.int32)
    for k, tk in enumerate(terms):
      acnt[k] = len(tk)
      for t, (mom, pdiff, coef) in enumerate(tk):
        amom[k, t] = mom; acoef[k, t] = coef
        for dd in range(d):
          apd[dd][k, t] = pdiff[dd]
    return amom, apd, acoef, acnt, aord, mono_cmidx

  def _oi_recipe(self):
    """Mesh-independent oscillation-matrix recipe (Eq 23), dimension-generic: sum
    over derivative multi-indices p (1 <= |p| <= order) of the products of the two
    differentiated basis monomials, expressed via the central moments."""
    from itertools import product

    def ff(n, k):
      r = 1.0
      for j in range(k):
        r *= (n - j)
      return r

    d = self.dim
    oik, oiq, oimom, oicoef, oiord = [], [], [], [], []
    for p in product(range(self.order + 1), repeat=d):
      sp = sum(p)
      if sp < 1 or sp > self.order:
        continue
      for k, ek in enumerate(self.exps):
        if any(pi > eki for pi, eki in zip(p, ek)):
          continue
        ck = 1.0
        for eki, pi in zip(ek, p):
          ck *= ff(eki, pi)
        for kq, eq in enumerate(self.exps):
          if any(pi > eqi for pi, eqi in zip(p, eq)):
            continue
          cq = 1.0
          for eqi, pi in zip(eq, p):
            cq *= ff(eqi, pi)
          A = tuple((eki - pi) + (eqi - pi) for eki, eqi, pi in zip(ek, eq, p))
          oik.append(k); oiq.append(kq); oimom.append(self._cm_idx[A])
          oicoef.append(ck * cq); oiord.append(sum(A))
    return (np.array(oik, np.int32), np.array(oiq, np.int32), np.array(oimom, np.int32),
            np.array(oicoef), np.array(oiord, np.int32))

  def _extend_field(self, U):
    """Return U extended with the halo (other-rank) cell values, so the compiled
    kernels can index the halo neighbours that a partition-crossing stencil selects.
    In serial (nh == 0) it returns U unchanged. The halo values are exchanged from
    the *current* U each call, so it is always up to date within an RK stage."""
    U = np.ascontiguousarray(U, dtype=float)
    if self.nh == 0:
      return U
    self.domain.halo_comm.exchange(U, recv_buffer=self._uhalo)
    Ue = np.empty(self.nb + self.nh)
    Ue[:self.nb] = U
    Ue[self.nb:] = self._uhalo
    return Ue

  def exchange_coeffs(self, coeffs):
    """Exchange per-cell reconstruction coefficients (nb, K) to this rank's halo,
    returning (nh, K). Used to evaluate a halo neighbour's WENO polynomial at a
    partition face for a full-order flux there. Empty in serial."""
    out = np.zeros((self.nh, self.K))
    if self.nh == 0:
      return out
    buf = np.zeros(self.nh)
    for k in range(self.K):
      self.domain.halo_comm.exchange(np.ascontiguousarray(coeffs[:, k]), recv_buffer=buf)
      out[:, k] = buf
    return out

  def halo_face_values(self, coeffs_h, Uh, hf, fcx, fcy, fcz=None):
    """Evaluate the halo neighbours' WENO polynomials at partition-face centres.
    coeffs_h (nh,K), Uh (nh,) halo cell values, hf face->halo ids, fcx/fcy(/fcz)
    the face centres of those faces. Returns the reconstructed face values."""
    hxr = (fcx - self._cx_h[hf]) / self._h_h[hf]
    hyr = (fcy - self._cy_h[hf]) / self._h_h[hf]
    phi = hxr[:, None] ** self._ea[None, :] * hyr[:, None] ** self._eb[None, :]
    if self.dim == 3:
      hzr = (fcz - self._cz_h[hf]) / self._h_h[hf]
      phi = phi * hzr[:, None] ** self._ec[None, :]
    phi = phi - self._M0_h[hf]
    return Uh[hf] + np.einsum('fk,fk->f', coeffs_h[hf], phi)

  def reconstruct(self, U):
    """k-exact reconstruction on the **central** stencil only (linear; high order
    in smooth regions, oscillatory at discontinuities). Returns coeffs (nbcells, K)."""
    Ue = self._extend_field(U)
    coeffs = np.zeros((self.nbcells, self.K))
    for i in range(self.nbcells):
      cnt = self._st_cnt[i, 0]
      st = self._st_idx[i, 0, :cnt]
      coeffs[i] = self._pinv_p[i, 0, :, :cnt] @ (Ue[st] - Ue[i])
    return coeffs

  def weno_reconstruct(self, U):
    """Non-linear WENO reconstruction (Eqs 17-23): blend the central and directional
    stencil polynomials with weights w_s = lam_s/(eps+SI_s)^power, normalised. In
    smooth regions the large central linear weight dominates (high order); near a
    discontinuity the stencils that cross it get a large SI and are suppressed, so
    the reconstruction stays essentially non-oscillatory. Returns coeffs (nbcells, K).

    Runs the compiled (numba) hot loop over the precomputed mesh-dependent arrays."""
    Ue = self._extend_field(U)
    coeffs = np.zeros((self.nbcells, self.K))
    self._kernel(Ue, coeffs, self._st_idx, self._st_cnt, self._pinv_p,
                 self._OI_p, self._lam_arr, self.eps, self.power)
    return coeffs

  def advect_residual(self, u_cell, u_ghost, coeffs, ax, ay):
    """WENO linear-advection residual d(u*vol)/dt for velocity (ax, ay):
    upwind flux with the WENO polynomial evaluated at face centres."""
    rez = np.zeros(self.nbcells)
    self._adv_kernel(rez, np.ascontiguousarray(u_cell, dtype=float),
                     np.ascontiguousarray(u_ghost, dtype=float), coeffs,
                     self._ea, self._eb, self._M0_p, self._cx, self._cy, self.h,
                     self._fcx, self._fcy, self.domain.faces.cellid,
                     self.domain.faces.normal, self.domain.faces.mesure,
                     np.asarray(self.domain.faces.name, dtype=np.uint32), float(ax), float(ay))
    return rez

  def smoothness(self, coeffs):
    """Smoothness indicator SI_i = a_i^T OI_i a_i for every cell."""
    return np.einsum('ik,ikq,iq->i', coeffs, self._OI_p, coeffs)

  def evaluate(self, U, coeffs, i, *xp):
    """Evaluate the cell-i reconstruction polynomial at a physical point (x, y[, z])."""
    hi = self.h[i]
    ci = self.center[i]
    val = U[i]
    for k, e in enumerate(self.exps):
      phi = 1.0
      for dd, a in enumerate(e):
        phi *= ((xp[dd] - ci[dd]) / hi) ** a
      val += coeffs[i, k] * (phi - self._M0_p[i][k])
    return val
