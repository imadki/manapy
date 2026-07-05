import os
import numpy as np
import numpy.typing as npt
import meshio
from mpi4py import MPI
import manapy.backends.types as types
import shutil
import threading
import time

class VTKWriter:
    def __init__(self, nodes, dim, cells, cell_type, comm=MPI.COMM_WORLD, vtk_precision="Float64"):
        self.nodes = nodes
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.vtk_precision = vtk_precision

        self.cell_type = cell_type
        self.cells = self._get_cells_tuple(dim, cells, cell_type)
        self.vtk_path = self._get_vtk_path(self.rank)
        self.nbnodes = len(nodes.vertex)
        self.nbcells = len(cells)

    @staticmethod
    def _get_vtk_path(rank):
        # Supprimer les milliers de vtk du run precedent sur NFS prend ~10 s et
        # etait paye DANS Domain.__init__ a chaque run (compte dans "Time to
        # create the domain") : on renomme (une seule op metadata) et on
        # supprime dans un thread d'arriere-plan. Les restes d'un run tue
        # (vtk_results.old.*) sont balayes au passage.
        vtkpath = "vtk_results"
        if rank == 0:
            stale = [d for d in os.listdir('.') if d.startswith(vtkpath + ".old.")]
            if os.path.exists(vtkpath):
                old = f"{vtkpath}.old.{os.getpid()}_{int(time.time() * 1e6)}"
                os.rename(vtkpath, old)
                stale.append(old)
            if stale:
                threading.Thread(target=VTKWriter._rmtree_many, args=(stale,), daemon=True).start()
            os.mkdir(vtkpath)
        return vtkpath

    @staticmethod
    def _rmtree_many(paths):
        for p in paths:
            shutil.rmtree(p, ignore_errors=True)

    @staticmethod
    def _get_cells_tuple(dim, cells, cells_type):
        res = []

        if dim == 2:
            quads = cells[cells_type == types.MeshCell.QUAD][:, :4]
            if len(quads) > 0: res.append(("quad", quads))

            triangles = cells[cells_type == types.MeshCell.TRIANGLE][:, :3]
            if len(triangles) > 0: res.append(("triangle", triangles))
        elif dim == 3:
            tetras = cells[cells_type == types.MeshCell.TETRA][:, :4]
            if len(tetras) > 0: res.append(("tetra", tetras))

            hexahedrons = cells[cells_type == types.MeshCell.HEXAHEDRON][:, :8]
            if len(hexahedrons) > 0: res.append(("hexahedron", hexahedrons))

            pyramids = cells[cells_type == types.MeshCell.PYRAMID][:, :5]
            if len(pyramids) > 0: res.append(("pyramid", pyramids))
        return res


    @staticmethod
    def _as_host_array(value):
        if hasattr(value, "to_host"):
            return np.asarray(value.to_host())
        return np.asarray(value)

    # --------------------------------------------------
    def _log(self, niter: int, time: int, dt: int, variables: list[str], values: list[npt.NDArray]):
        # Compute max for each variable
        maxvals = {}
        for var, val in zip(variables, values):
            local_max = np.array([np.max(self._as_host_array(val))], dtype=np.float64)
            global_max = np.zeros(1, dtype=np.float64)
            self.comm.Reduce(local_max, global_max, op=MPI.MAX, root=0)
            maxvals[var] = global_max[0]

        # Logging
        if self.rank == 0:
            print("*************** Saving Results ***************")
            print(f"Iteration = {niter}, time = {time}, dt = {dt}")
            for var in variables:
                print(f"max({var}) = {maxvals[var]}")

    # --------------------------------------------------
    def _write_vtu(self, filename, points, point_data=None, cell_data=None):
        filepath = os.path.join(self.vtk_path, filename)

        meshio.write_points_cells(
            filepath,
            points,
            self.cells,
            point_data=point_data,
            cell_data=cell_data,
            file_format="vtu"
        )

    # --------------------------------------------------
    # FIX 3: Correct cell_data structure for meshio
    # --------------------------------------------------
    def _format_cell_data(self, variables: list[str], values: list[npt.NDArray]):
        """
        meshio requires:
        cell_data = {name: [array_block1, array_block2, ...]}
        aka [[values_for_triangles], [values_for_quads], ...] same order in self.cells
        """
        cell_type_dic = {
            "quad": types.MeshCell.QUAD,
            "triangle": types.MeshCell.TRIANGLE,
            "tetra": types.MeshCell.TETRA,
            "hexahedron": types.MeshCell.HEXAHEDRON,
            "pyramid": types.MeshCell.PYRAMID
        }
        formatted = {}

        for i in range(len(variables)):
            name = variables[i]
            data = self._as_host_array(values[i])

            split_data = []

            for cell_type_name, _ in self.cells:
                block_data = data[self.cell_type == cell_type_dic[cell_type_name]]
                split_data.append(block_data)

            formatted[name] = split_data

        return formatted


    # --------------------------------------------------
    def _write_pvtu(self, miter, variables, location="point"):
        if self.rank != 0:
            return

        path = os.path.join(self.vtk_path, f"visu{miter}.pvtu")

        with open(path, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
            f.write('<PUnstructuredGrid GhostLevel="0">\n')

            # Points
            f.write('<PPoints>\n')
            f.write(f'<PDataArray type="{self.vtk_precision}" NumberOfComponents="3"/>\n')
            f.write('</PPoints>\n')

            # Cells (VTK-correct types)
            f.write('<PCells>\n')
            f.write('<PDataArray type="Int32" Name="connectivity"/>\n')
            f.write('<PDataArray type="Int32" Name="offsets"/>\n')
            f.write('<PDataArray type="UInt8" Name="types"/>\n')
            f.write('</PCells>\n')

            # Data
            tag = "PPointData" if location == "point" else "PCellData"
            f.write(f'<{tag}>\n')
            for var in variables:
                f.write(f'<PDataArray type="{self.vtk_precision}" Name="{var}"/>\n')
            f.write(f'</{tag}>\n')

            # Pieces
            for i in range(self.size):
                fname = f"visu{i}-{miter}.vtu"
                f.write(f'<Piece Source="{fname}"/>\n')

            f.write('</PUnstructuredGrid>\n')
            f.write('</VTKFile>\n')

    # ==================================================
    # PUBLIC METHODS
    # ==================================================

    def save_node_multi(self, variables: list[str], values: list[npt.NDArray], miter: int, niter: int, time: int, dt: int):
        if len(variables) == 0 or len(values) == 0:
            return
        if len(variables) != len(values):
            raise ValueError("mismatched length")
        for item in values:
            assert len(item) == self.nbnodes

        self._log(niter, time, dt, variables, values)
        points = np.asarray(self.nodes.vertex[:, :3], dtype=types.np_float_type)

        point_data = {var: self._as_host_array(val) for var, val in zip(variables, values)}
        fname = f"visu{self.rank}-{miter}.vtu"
        self._write_vtu(fname, points, point_data=point_data)
        self._write_pvtu(miter, variables, "point")

    # --------------------------------------------------

    def save_cell_multi(self, variables: list[str], values: list[npt.NDArray], miter: int, niter: int, time: int, dt: int):
        if len(variables) == 0 or len(values) == 0:
            return
        if len(variables) != len(values):
            raise ValueError("mismatched length")
        for item in values:
            assert len(item) == self.nbcells

        self._log(niter, time, dt, variables, values)
        points = np.asarray(self.nodes.vertex[:, :3], dtype=types.np_float_type)

        cell_data = self._format_cell_data(variables, values)
        fname = f"visu{self.rank}-{miter}.vtu"
        self._write_vtu(fname, points, cell_data=cell_data)
        self._write_pvtu(miter, variables, "cell")