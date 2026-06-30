#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba kernels for passive multispecies transport on the unstructured FV mesh.

Each species partial density q_k = rho * Y_k is advected by the resolved flow
with a Rusanov flux that mirrors the bulk density flux exactly (same wave speed
S, same upwind structure), so the species transport is consistent with the
compressible Euler/NS solver it rides on. A Fickian diffusion kernel adds
rho * D_k * grad(Y_k). Reaction source terms are added later (Phase 5).

Nothing is compiled at import; call setup(dim) once on every rank.
"""
from manapy.backends.compile_fun import compile
import numpy as np

_done = False


def _explicitscheme_species_2d(rez: 'float64[:]', q_c: 'float64[:]', q_h: 'float64[:]',
                               rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]',
                               rho_h: 'float64[:]', P_h: 'float64[:]', rhou_h: 'float64[:]', rhov_h: 'float64[:]',
                               cellidf: 'int32[:,:]', halofid: 'int32[:]',
                               normal: 'float64[:,:]', mesure: 'float64[:]', name: 'uint32[:]', gamma: 'float64'):
    # Rusanov advection of a species partial density q = rho*Y, consistent with
    # the bulk Euler density flux. Boundary faces use a zero-gradient outer state.
    rez[:] = np.zeros(len(rez))
    nbface = len(cellidf)
    for i in range(nbface):
        mes = mesure[i]
        nx = normal[i][0] / mes
        ny = normal[i][1] / mes
        il = cellidf[i][0]
        rhoL = rho_c[il]
        uL = rhou_c[il] / rhoL
        vL = rhov_c[il] / rhoL
        PL = P_c[il]
        qL = q_c[il]
        unL = uL * nx + vL * ny
        cL = np.sqrt(gamma * PL / rhoL)

        inner = name[i] == 0
        if inner:
            ir = cellidf[i][1]
            rhoR = rho_c[ir]
            uR = rhou_c[ir] / rhoR
            vR = rhov_c[ir] / rhoR
            PR = P_c[ir]
            qR = q_c[ir]
        elif name[i] == 10:
            h = halofid[i]
            rhoR = rho_h[h]
            uR = rhou_h[h] / rhoR
            vR = rhov_h[h] / rhoR
            PR = P_h[h]
            qR = q_h[h]
        else:
            rhoR = rhoL; uR = uL; vR = vL; PR = PL; qR = qL

        unR = uR * nx + vR * ny
        cR = np.sqrt(gamma * PR / rhoR)
        sL = np.fabs(unL) + cL
        sR = np.fabs(unR) + cR
        S = sL if sL > sR else sR

        fl = qL * unL
        fr = qR * unR
        flux = (0.5 * (fl + fr) - 0.5 * S * (qR - qL)) * mes
        rez[il] -= flux
        if inner:
            rez[cellidf[i][1]] += flux


def _explicitscheme_species_3d(rez: 'float64[:]', q_c: 'float64[:]', q_h: 'float64[:]',
                               rho_c: 'float64[:]', P_c: 'float64[:]', rhou_c: 'float64[:]', rhov_c: 'float64[:]', rhow_c: 'float64[:]',
                               rho_h: 'float64[:]', P_h: 'float64[:]', rhou_h: 'float64[:]', rhov_h: 'float64[:]', rhow_h: 'float64[:]',
                               cellidf: 'int32[:,:]', halofid: 'int32[:]',
                               normal: 'float64[:,:]', mesure: 'float64[:]', name: 'uint32[:]', gamma: 'float64'):
    rez[:] = np.zeros(len(rez))
    nbface = len(cellidf)
    for i in range(nbface):
        mes = mesure[i]
        nx = normal[i][0] / mes
        ny = normal[i][1] / mes
        nz = normal[i][2] / mes
        il = cellidf[i][0]
        rhoL = rho_c[il]
        uL = rhou_c[il] / rhoL
        vL = rhov_c[il] / rhoL
        wL = rhow_c[il] / rhoL
        PL = P_c[il]
        qL = q_c[il]
        unL = uL * nx + vL * ny + wL * nz
        cL = np.sqrt(gamma * PL / rhoL)

        inner = name[i] == 0
        if inner:
            ir = cellidf[i][1]
            rhoR = rho_c[ir]
            uR = rhou_c[ir] / rhoR
            vR = rhov_c[ir] / rhoR
            wR = rhow_c[ir] / rhoR
            PR = P_c[ir]
            qR = q_c[ir]
        elif name[i] == 10:
            h = halofid[i]
            rhoR = rho_h[h]
            uR = rhou_h[h] / rhoR
            vR = rhov_h[h] / rhoR
            wR = rhow_h[h] / rhoR
            PR = P_h[h]
            qR = q_h[h]
        else:
            rhoR = rhoL; uR = uL; vR = vL; wR = wL; PR = PL; qR = qL

        unR = uR * nx + vR * ny + wR * nz
        cR = np.sqrt(gamma * PR / rhoR)
        sL = np.fabs(unL) + cL
        sR = np.fabs(unR) + cR
        S = sL if sL > sR else sR

        fl = qL * unL
        fr = qR * unR
        flux = (0.5 * (fl + fr) - 0.5 * S * (qR - qL)) * mes
        rez[il] -= flux
        if inner:
            rez[cellidf[i][1]] += flux


def _explicitscheme_diffusion(rez: 'float64[:]', gx: 'float64[:]', gy: 'float64[:]', gz: 'float64[:]',
                              coef_f: 'float64[:]', cellidf: 'int32[:,:]',
                              normal: 'float64[:,:]', name: 'uint32[:]', dim: 'int32'):
    # Fickian diffusion flux of a scalar: G = coef_f * (grad(scalar).n), normal is
    # area-scaled. coef_f carries (rho*D) at the face. Conservative: +G to the
    # owner cell, -G to the neighbour (down-gradient diffusion).
    rez[:] = np.zeros(len(rez))
    nbface = len(cellidf)
    for i in range(nbface):
        g = gx[i] * normal[i][0] + gy[i] * normal[i][1]
        if dim == 3:
            g += gz[i] * normal[i][2]
        G = coef_f[i] * g
        il = cellidf[i][0]
        rez[il] += G
        if name[i] == 0:
            rez[cellidf[i][1]] -= G


def _update_species(q_c: 'float64[:]', rez: 'float64[:]', dtime: 'float64', vol: 'float64[:]'):
    q_c[:] += dtime * (rez[:] / vol[:])


def setup(dim=2):
    global _done
    if _done:
        return
    global explicitscheme_species_2d, explicitscheme_species_3d, update_species
    global explicitscheme_diffusion
    explicitscheme_species_2d = compile(_explicitscheme_species_2d)
    explicitscheme_species_3d = compile(_explicitscheme_species_3d)
    explicitscheme_diffusion = compile(_explicitscheme_diffusion)
    update_species = compile(_update_species)
    _done = True
