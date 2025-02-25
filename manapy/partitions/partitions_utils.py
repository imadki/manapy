#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 11:24:48 2025

@author: kissami
"""

import numpy as np
from numpy import uint32, int32

# from manapy.ddm.ddm_utils3d import split_to_tetra
from manapy.ddm.ddm_utils2d import split_to_triangle


def define_ghost_node(mesh, periodic, nbnodes, dim):
  ghost_nodes = np.zeros(nbnodes, dtype=np.int32)

  if dim == 2:
    typefaces = ["line"]
  elif dim == 3:
    typefaces = ["quad", "triangle"]

  ghost = {}
  if type(mesh.cells) == dict:
    for i, j in mesh.cell_data.items():
      if i in typefaces:
        ghost[i] = j.get('gmsh:physical')

    for i, j in mesh.cells.items():
      if i in typefaces:
        for k in range(len(ghost[i])):
          for index in range(len(j[k])):
            if ghost[i][k] == 1 or ghost[i][k] == 2:  # or ghost[k] == 5 or ghost[k] == 6 :
              ghost_nodes[j[k][index]] = int(ghost[i][k])

      if i in typefaces:
        for k in range(len(ghost[i])):
          for index in range(len(j[k])):
            if ghost_nodes[j[k][index]] != 1 and ghost_nodes[j[k][index]] != 2:
              if ghost[i][k] == 3 or ghost[i][k] == 4:  # or ghost[k] == 5 or ghost[k] == 6 :
                ghost_nodes[j[k][index]] = int(ghost[i][k])

    for i, j in mesh.cells.items():
      if i in typefaces:
        for k in range(len(ghost[i])):
          for index in range(len(j[k])):
            if ghost_nodes[j[k][index]] == 0:
              ghost_nodes[j[k][index]] = int(ghost[i][k])

  if periodic[0] == 1:
    for i in range(len(ghost_nodes)):
      if ghost_nodes[i] == 1:
        ghost_nodes[i] = 11
      elif ghost_nodes[i] == 2:
        ghost_nodes[i] = 22

  if periodic[1] == 1:
    for i in range(len(ghost_nodes)):
      if ghost_nodes[i] == 3:
        ghost_nodes[i] = 33
      elif ghost_nodes[i] == 4:
        ghost_nodes[i] = 44

  if periodic[2] == 1:
    for i in range(len(ghost_nodes)):
      if ghost_nodes[i] == 5:
        ghost_nodes[i] = 55
      elif ghost_nodes[i] == 6:
        ghost_nodes[i] = 66

  return ghost_nodes


def convert_2d_cons_to_array(ele1: 'uint32[:,:]', ele2: 'uint32[:,:]'):
  nbelements = 0
  if len(ele1) > 1:
    nbelements += len(ele1)
    l = 3

  if len(ele2) > 1:
    nbelements += len(ele2)
    l = 4

  padded_l = np.zeros((nbelements, l + 1), dtype=np.int32)

  if len(ele1) > 1:
    for i in range(len(ele1)):
      padded_l[i][0:3] = ele1[i][0:3]
      padded_l[i][-1] = 3

    for i in range(len(ele1), nbelements):
      padded_l[i][0:4] = ele2[i - len(ele1)]
      padded_l[i][-1] = 4

  else:
    for i in range(nbelements):
      padded_l[i][0:4] = ele2[i]
      padded_l[i][-1] = 4

  return padded_l


def convert_3d_cons_to_array(ele1: 'uint32[:,:]', ele2: 'uint32[:,:]', ele3: 'uint32[:,:]'):
  nbelements = 0

  # tetra
  if len(ele1) > 1:
    nbelements += len(ele1)
    l = 4

  # pyramid
  if len(ele2) > 1:
    nbelements += len(ele2)
    l = 5

  # hexa
  if len(ele3) > 1:
    nbelements += len(ele3)
    l = 8

  padded_l = np.zeros((nbelements, l + 1), dtype=np.int32)

  if len(ele1) > 1:
    for i in range(len(ele1)):
      padded_l[i][0:4] = ele1[i][0:4]
      padded_l[i][-1] = 4

    for i in range(len(ele1), nbelements - len(ele3)):
      padded_l[i][0:5] = ele2[i - len(ele1)]
      padded_l[i][-1] = 5

    for i in range(len(ele1) + len(ele2), nbelements):
      padded_l[i][0:8] = ele3[i - (len(ele1) + len(ele2))]
      padded_l[i][-1] = 8

  elif len(ele2) > 1:
    for i in range(nbelements):
      padded_l[i][0:5] = ele2[i]
      padded_l[i][-1] = 5

  elif len(ele3) > 1:
    for i in range(nbelements):
      padded_l[i][0:8] = ele3[i]
      padded_l[i][-1] = 8

  return padded_l


def create_npart_cpart(cell_nodeid: 'uint32[:,:]', npart: 'uint32[:]',
                       epart: 'uint32[:]', nbnodes: 'int32', nbelements: 'int32',
                       SIZE: 'int32'):
  npart = [[i] for i in npart]
  cpart = [[i] for i in epart]
  neighsub = [[i for i in range(0)] for i in range(SIZE)]
  globcelltoloc = [[i for i in range(0)] for i in range(SIZE)]
  locnodetoglob = [[i for i in range(0)] for i in range(SIZE)]
  halo_cellid = [[i for i in range(0)] for i in range(SIZE)]

  for i in range(nbelements):
    for j in range(cell_nodeid[i][-1]):
      k = cell_nodeid[i][j]
      if epart[i] not in npart[k]:
        npart[k].append(epart[i])
      locnodetoglob[epart[i]].append(k)
    globcelltoloc[epart[i]].append(i)

  for i in range(nbelements):
    for j in range(cell_nodeid[i][-1]):
      for k in range(len(npart[cell_nodeid[i][j]])):
        if npart[cell_nodeid[i][j]][k] not in cpart[i]:
          cpart[i].append(npart[cell_nodeid[i][j]][k])

  maxnpart = 0
  for i in range(nbnodes):
    maxnpart = max(maxnpart, len(npart[i]))
    for j in range(len(npart[i])):
      neighsub[npart[i][j]].extend(npart[i])

  for i in range(SIZE):
    for cell in globcelltoloc[i]:
      # check if the cell's partition belong to two subdomains
      if len(cpart[cell]) > 1:
        # append the cell's partition to halo_cellid list
        halo_cellid[i].append(cell)

  npartnew = -1 * np.ones((nbnodes, maxnpart + 2), dtype=np.int32)
  # convert npart to array
  for i in range(nbnodes):
    for j in range(len(npart[i])):
      npartnew[i][j] = npart[i][j]
    npartnew[i][-2] = len(npart[i])
    npartnew[i][-1] = i

  tc = np.zeros(nbelements, dtype=np.uint32)
  cmpt = 0
  for i in range(SIZE):
    for j in range(len(globcelltoloc[i])):
      tc[cmpt] = globcelltoloc[i][j]
      cmpt += 1

  return npartnew, cpart, neighsub, halo_cellid, globcelltoloc, locnodetoglob, tc


def _compute_2d_halo(haloint, centvol, haloextloc, halointloc, halointlen, SIZE, neighsub,
                     cell_nodeid, nodes):
  vertices = np.zeros((4, 2))
  triangles = np.zeros((4, 3, 2))

  for i in range(SIZE):
    for j in range(len(neighsub[i])):
      halointloc[i].extend(haloint[(i, neighsub[i][j])])
      halointlen[i].append(len(haloint[(i, neighsub[i][j])]))

      for k in haloint[(neighsub[i][j], i)]:
        n_vertices = cell_nodeid[k][-1]
        cell_vertices = cell_nodeid[k][:n_vertices]

        if n_vertices == 3:  # Triangle
          coords = nodes[cell_vertices, :2]
          x1, y1 = coords[0]
          x2, y2 = coords[1]
          x3, y3 = coords[2]

          center = (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3
          volume = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

          centvol[i].append([*center, 0.0, volume])
          haloextloc[i].append([k, *cell_vertices, -1, 4])

        elif n_vertices == 4:  # Quadrilateral
          coords = nodes[cell_vertices, :2]
          vertices[:4] = coords
          split_to_triangle(vertices, triangles)

          center = triangles[0][2]
          volume = sum(
            0.5 * abs((tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) -
                      (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1]))
            for tri in triangles
          )

          centvol[i].append([*center, 0.0, volume])
          haloextloc[i].append([k, *cell_vertices, -1, 5])


def _compute_3d_halo(haloint, centvol, haloextloc, halointloc, halointlen, SIZE, neighsub,
                     cell_nodeid, nodes, precision):
  """Compute halo information for 3D cells."""
  wedge = np.zeros(3, dtype=precision)
  u = np.zeros(3, dtype=precision)
  v = np.zeros(3, dtype=precision)
  w = np.zeros(3, dtype=precision)

  for i in range(SIZE):
    for j in range(len(neighsub[i])):
      halointloc[i].extend(haloint[(i, neighsub[i][j])])
      halointlen[i].append(len(haloint[(i, neighsub[i][j])]))

      for k in haloint[(neighsub[i][j], i)]:
        n_vertices = cell_nodeid[k][-1]
        cell_vertices = cell_nodeid[k][:n_vertices]
        coords = nodes[cell_vertices, :3]

        if n_vertices == 4:  # Tetrahedron
          u = coords[1] - coords[0]
          v = coords[2] - coords[0]
          w = coords[3] - coords[0]

          wedge = np.cross(v, w)
          volume = abs(np.dot(u, wedge)) / 6.0

          center = coords.mean(axis=0)
          centvol[i].append([*center, volume])
          haloextloc[i].append([k, *cell_vertices, -1, -1, -1, -1, 5])

        elif n_vertices == 5:  # Pyramid
          # Manually split the pyramid into 4 tetrahedra
          tetrahedra = [
            [coords[0], coords[1], coords[2], coords[4]],  # First tetrahedron
            [coords[0], coords[2], coords[3], coords[4]]  # Second tetrahedron
          ]

          volume = 0.0
          for tetra in tetrahedra:
            u = tetra[1] - tetra[0]
            v = tetra[2] - tetra[0]
            w = tetra[3] - tetra[0]

            wedge = np.cross(v, w)
            volume += abs(np.dot(u, wedge)) / 6.0

          center = coords.mean(axis=0)
          centvol[i].append([*center, volume])
          haloextloc[i].append([k, *cell_vertices, -1, -1, -1, 6])

        elif n_vertices == 8:  # Hexahedron
          # Manually split the hexahedron into 6 tetrahedra
          tetrahedra = [
            [coords[0], coords[1], coords[3], coords[4]],
            [coords[1], coords[3], coords[4], coords[5]],
            [coords[4], coords[5], coords[3], coords[7]],
            [coords[1], coords[3], coords[5], coords[2]],
            [coords[3], coords[7], coords[5], coords[2]],
            [coords[5], coords[7], coords[6], coords[2]]
          ]

          volume = 0.0
          for tetra in tetrahedra:
            u = tetra[1] - tetra[0]
            v = tetra[2] - tetra[0]
            w = tetra[3] - tetra[0]

            wedge = np.cross(v, w)
            volume += abs(np.dot(u, wedge)) / 6.0

          center = coords.mean(axis=0)
          centvol[i].append([*center, volume])
          haloextloc[i].append([k, *cell_vertices, 9])


def compute_halocell(halo_cellid, cpart, cell_nodeid, nodes, neighsub, SIZE, dim, precision):
  haloint = {}
  for i in range(SIZE):
    for cell in halo_cellid[i]:
      for k in range(len(cpart[cell])):
        if i != cpart[cell][k]:
          haloint.setdefault((i, cpart[cell][k]), []).append(cell)
  centvol = [[] for i in range(SIZE)]
  haloextloc = [[] for i in range(SIZE)]
  halointloc = [[] for i in range(SIZE)]
  halointlen = [[] for i in range(SIZE)]

  if dim == 2:
    _compute_2d_halo(haloint, centvol, haloextloc, halointloc, halointlen, SIZE,
                     neighsub, cell_nodeid, nodes)
  elif dim == 3:
    _compute_3d_halo(haloint, centvol, haloextloc, halointloc, halointlen, SIZE,
                     neighsub, cell_nodeid, nodes, precision)

  return centvol, haloextloc, halointloc, halointlen