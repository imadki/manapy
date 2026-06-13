def _facetocell(u_face: 'float[:]', u_c: 'float[:]', cell_faceid: 'int[:,:]', dim: 'int'):
  nbelements = len(u_c)

  for i in range(nbelements):
    u_c[i] = 0.0
    for j in range(cell_faceid[i][-1]):
      u_c[i] += u_face[cell_faceid[i][j]]

  for i in range(nbelements):
    u_c[i] /= cell_faceid[i][-1]


def _celltoface(u_cell: 'float[:]', u_face: 'float[:]', u_ghost: 'float[:]', u_halo: 'float[:]',
               face_cellid: 'int[:,:]', face_halofid: 'int[:]',
               d_innerfaces: 'int[:]', d_boundaryfaces: 'int[:]', d_halofaces: 'int[:]'):
  for i in d_innerfaces:
    c1 = face_cellid[i][0]
    c2 = face_cellid[i][1]
    u_face[i] = .5 * (u_cell[c1] + u_cell[c2])

  for i in d_halofaces:
    c1 = face_cellid[i][0]
    u_face[i] = .5 * (u_cell[c1] + u_halo[face_halofid[i]])

  for i in d_boundaryfaces:
    c1 = face_cellid[i][0]
    u_face[i] = .5 * (u_cell[c1] + u_ghost[i])

