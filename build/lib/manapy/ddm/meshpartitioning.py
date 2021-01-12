# coding: utf-8
import os
import timeit
from collections import OrderedDict
import meshio
import numpy as np
from mgmetis import metis

__all__ = ['meshpart']

def meshpart(size, filename):
    if size > 1:
        paramesh(size, filename)
    else:
        seqmesh(filename)

def seqmesh(filename):

    def load_gmsh_mesh(filename):
        #mesh = meshio.gmsh.read(filename)
        mesh = meshio.read(filename)
        return mesh

    def create_cell_nodeid(mesh):
        cell_nodeid = []

        if type(mesh.cells) == dict:
            cell_nodeid = mesh.cells["triangle"]
        elif type(mesh.cells) == list:
            cell_nodeid = mesh.cells[1].data

        for i in range(len(cell_nodeid)):
            cell_nodeid[i].sort()

        return cell_nodeid

    def define_ghost_node(mesh, nodes):
        ghost_nodes = [0]*len(nodes)

        if type(mesh.cells) == dict:
            for i, j in mesh.cell_data.items():
                if i == "line":
                    ghost = j.get('gmsh:physical')

            for i, j in mesh.cells.items():
                if i == "line":
                    for k in range(len(j)):
                        for index in range(2):
                            if ghost[k] > 2:
                                ghost_nodes[j[k][index]] = int(ghost[k])
            for i, j in mesh.cells.items():
                if i == "line":
                    for k in range(len(j)):
                        for index in range(2):
                            if ghost[k] <= 2:
                                ghost_nodes[j[k][index]] = int(ghost[k])

        elif type(mesh.cells) == list:
            ghost = mesh.cell_data['gmsh:physical'][0]
            for i in range(len(mesh.cells[0].data)):
                for j in range(2):
                    if ghost[i] > 2:
                        ghost_nodes[mesh.cells[0].data[i][j]] = int(ghost[i])

            for i in range(len(mesh.cells[0].data)):
                for j in range(2):
                    if ghost[i] <= 2:
                        ghost_nodes[mesh.cells[0].data[i][j]] = int(ghost[i])

        return ghost_nodes

    def create_nodes(mesh):
        nodes = []
        nodes = mesh.points
        return nodes

    start = timeit.default_timer()

    #load mesh
    mesh = load_gmsh_mesh(filename)
    #coordinates x, y of each node
    nodes = create_nodes(mesh)
    #nodes of each cell
    cell_nodeid = create_cell_nodeid(mesh)

    ghost_nodes = define_ghost_node(mesh, nodes)

    if os.path.exists("mesh"+str(0)+".txt"):
        os.remove("mesh"+str(0)+".txt")

    with open("mesh"+str(0)+".txt", "a") as text_file:
        text_file.write("elements\n")
        np.savetxt(text_file, cell_nodeid, fmt='%u')
        text_file.write("endelements\n")


    with open("mesh"+str(0)+".txt", "a") as text_file:
        text_file.write("nodes\n")
        for i in range(len(nodes)):
            for j in range(3):
                text_file.write(str(nodes[i][j])+str(" "))
            text_file.write(str(ghost_nodes[i]))
            text_file.write("\n")
        text_file.write("endnodes\n")

    stop = timeit.default_timer()

    print('Global Exxecution Time: ', stop - start)

def paramesh(size, filename):

    def load_gmsh_mesh(filename):
        mesh = meshio.read(filename)
        return mesh

    def create_cell_nodeid(mesh):
        cell_nodeid = []

        if type(mesh.cells) == dict:
            cell_nodeid = mesh.cells["triangle"]
        elif type(mesh.cells) == list:
            cell_nodeid = mesh.cells[1].data

        for i in range(len(cell_nodeid)):
            cell_nodeid[i].sort()

        return cell_nodeid

    def define_ghost_node(mesh, nodes):
        ghost_nodes = [0]*len(nodes)

        if type(mesh.cells) == dict:
            for i, j in mesh.cell_data.items():
                if i == "line":
                    ghost = j.get('gmsh:physical')

            for i, j in mesh.cells.items():
                if i == "line":
                    for k in range(len(j)):
                        for index in range(2):
                            if ghost[k] > 2:
                                ghost_nodes[j[k][index]] = int(ghost[k])
            for i, j in mesh.cells.items():
                if i == "line":
                    for k in range(len(j)):
                        for index in range(2):
                            if ghost[k] <= 2:
                                ghost_nodes[j[k][index]] = int(ghost[k])

        elif type(mesh.cells) == list:
            ghost = mesh.cell_data['gmsh:physical'][0]
            for i in range(len(mesh.cells[0].data)):
                for j in range(2):
                    if ghost[i] > 2:
                        ghost_nodes[mesh.cells[0].data[i][j]] = int(ghost[i])

            for i in range(len(mesh.cells[0].data)):
                for j in range(2):
                    if ghost[i] <= 2:
                        ghost_nodes[mesh.cells[0].data[i][j]] = int(ghost[i])

        return ghost_nodes

    start = timeit.default_timer()

    #load mesh
    mesh = load_gmsh_mesh(filename)

    #coordinates x, y of each node
    nodes = mesh.points#create_nodes(mesh.points)
    #nodes of each cell
    cell_nodeid = create_cell_nodeid(mesh)

    cell_nodeiddict = {tuple(cell_nodeid[0]): 0}
    for i in range(1, len(cell_nodeid)):
        cell_nodeiddict[tuple(cell_nodeid[i])] = i

    #ghost nodes
    ghost_nodes = define_ghost_node(mesh, nodes)

    stopmesh = timeit.default_timer()
    print("Reading mesh", stopmesh-start)

    nbelements = len(cell_nodeid)
    nbnodes = len(nodes)

    print("Number of Cells : ", nbelements)
    print("Number of Nodes : ", nbnodes)

    #Partitioning mesh
    if size > 1:
        objval, epart, npart = metis.part_mesh_dual(size, cell_nodeid)

    stopmetis = timeit.default_timer()
    print("METIS partitionning in ", size, "partitions", stopmetis - stopmesh)

    node_parts = OrderedDict()
    cell_parts = OrderedDict()
    node_part = [[]  for i in range(size)]
    cell_part = [[]  for i in range(size)]

    globnodetoloc = OrderedDict()
    locnodetoglob = OrderedDict()
    globcelltoloc = OrderedDict()

    neighsub = [[]  for i in range(size)]
    halo_cellid = [[]  for i in range(size)]
    npart = [[] for i in range(nbnodes)]
    cpart = [[] for i in range(nbelements)]

    for i in range(nbelements):
        for j in range(3):
            if epart[i] not in npart[cell_nodeid[i][j]]:
                npart[cell_nodeid[i][j]].append(epart[i])

    for i in range(nbelements):
        for j in range(3):
            for k in range(len(npart[cell_nodeid[i][j]])):
                if npart[cell_nodeid[i][j]][k] not in cpart[i]:
                    cpart[i].append(npart[cell_nodeid[i][j]][k])
        cpart[i].sort()

    #Create dict of nodes/cells for each partition
    for i in range(nbelements):
        for j in range(3):
            k = cell_nodeid[i][j]
            node_parts[epart[i], k] = [nodes[k][0], nodes[k][1], nodes[k][2], ghost_nodes[k]]
            cell_parts[epart[i], i] = cell_nodeid[i]

    #Create list of nodes/cells for each partition and local to global indexation
    for i, j in node_parts.items():
        node_part[i[0]].append(j)
        globnodetoloc[i[0], i[1]] = len(node_part[i[0]])-1
        locnodetoglob[i[0], len(node_part[i[0]])-1] = i[1]
        if len(npart[i[1]]) > 1:
            for index in range(len(npart[i[1]])):
                if (npart[i[1]][index] not in neighsub[i[0]] and npart[i[1]][index] != i[0]):
                    neighsub[i[0]].append(npart[i[1]][index])
                    neighsub[i[0]].sort()

    for i, j in cell_parts.items():
        cell_part[i[0]].append(j)
        globcelltoloc[i[0], len(cell_part[i[0]])-1] = i[1]
        #globcelltoloc[i[0],i[1]] = len(cell_part[i[0]])

    stopstruc = timeit.default_timer()
    print("Create local structure for each proc", stopstruc - stopmetis)

    for i in range(size):
        for j in cell_part[i]:
            if (len(npart[j[0]]) + len(npart[j[1]]) + len(npart[j[2]])) > 3:
                halo_cellid[i].append(j)

    haloint = OrderedDict()
    haloext = OrderedDict()
    for i in range(size):
        for j in halo_cellid[i]:
            cell = cell_nodeiddict.get(tuple(j))
            for k in range(len(cpart[cell])):
                if i != cpart[cell][k]:
                    haloint.setdefault((i, cpart[cell][k]), []).append(cell)
                    haloext.setdefault((cpart[cell][k], i), []).append(cell)

    for i in range(size):
        for j in range(len(cell_part[i])):
            cell_part[i][j] = [globnodetoloc[i, cell_part[i][j][0]],
                               globnodetoloc[i, cell_part[i][j][1]],
                               globnodetoloc[i, cell_part[i][j][2]]]

    stophalo = timeit.default_timer()
    print("Creating halo structure", stophalo - stopstruc)

    centvol = [[] for i in range(size)]
    for i in range(size):
        for j in range(len(neighsub[i])):
            for k in range(len(haloext[(i, neighsub[i][j])])):
                s_1 = cell_nodeid[haloext[(i, neighsub[i][j])][k]][0]
                s_2 = cell_nodeid[haloext[(i, neighsub[i][j])][k]][1]
                s_3 = cell_nodeid[haloext[(i, neighsub[i][j])][k]][2]

                x_1 = nodes[s_1][0]
                y_1 = nodes[s_1][1]
                x_2 = nodes[s_2][0]
                y_2 = nodes[s_2][1]
                x_3 = nodes[s_3][0]
                y_3 = nodes[s_3][1]

                centvol[i].append([1./3 * (x_1 + x_2 + x_3), 1./3*(y_1 + y_2 + y_3),
                                   (1./2) * abs((x_1-x_2)*(y_1-y_3)-(x_1-x_3)*(y_1-y_2))])

    for i in range(size):
        if os.path.exists("mesh"+str(i)+".txt"):
            os.remove("mesh"+str(i)+".txt")

    for i in range(size):
        with open("mesh"+str(i)+".txt", "a") as text_file:
            text_file.write("elements\n")
            np.savetxt(text_file, cell_part[i], fmt='%u')
            text_file.write("endelements\n")
            text_file.write("nodes\n")
            np.savetxt(text_file, node_part[i])
            text_file.write("endnodes\n")
            text_file.write("halosint\n")
            for j in range(len(neighsub[i])):
                for k in range(len(haloint[(i, neighsub[i][j])])):
                    text_file.write(str(haloint[(i, neighsub[i][j])][k]))
                    text_file.write("\n")
            text_file.write("endhalosint\n")
            text_file.write("halosext\n")
            for j in range(len(neighsub[i])):
                for k in range(len(haloext[(i, neighsub[i][j])])):
                    text_file.write(str(cell_nodeid[haloext[(i, neighsub[i][j])][k]][0])+" "+
                                    str(cell_nodeid[haloext[(i, neighsub[i][j])][k]][1])+" "+
                                    str(cell_nodeid[haloext[(i, neighsub[i][j])][k]][2]))
                    text_file.write("\n")
            text_file.write("endhalosext\n")
            text_file.write("centvol\n")
            np.savetxt(text_file, centvol[i])
            text_file.write("endcentvol\n")
            text_file.write("globalcelltolocal\n")
            for j in range(len(cell_part[i])):
                text_file.write(str(globcelltoloc[i, j]))
                text_file.write("\n")
            text_file.write("endglobalcelltolocal\n")
            text_file.write("localnodetoglobal\n")
            for j in range(len(node_part[i])):
                text_file.write(str(locnodetoglob[i, j]))
                text_file.write("\n")
            text_file.write("endlocalnodetoglobal\n")
            text_file.write("neigh\n")
            for j in range(len(neighsub[i])):
                text_file.write(str(neighsub[i][j])+ " ")
            text_file.write("\n")
            for j in neighsub[i]:
                text_file.write(str(len(haloint[(i, j)]))+ " ")
            text_file.write("\n")
            text_file.write("endneigh\n")

    stopfile = timeit.default_timer()
    print("save structures in files", stopfile - stophalo)

    stop = timeit.default_timer()
    print('Global Execution Time: ', stop - start)
