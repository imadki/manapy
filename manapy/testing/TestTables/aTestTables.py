from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

@dataclass(slots=True, init=False)
class ATestTables:
    width: int
    WIDTH: float
    HEIGHT: float
    DEPTH: float
    x_length: float
    y_length: float
    z_length: float

    nb_cells: int
    nb_nodes: int
    nb_faces: int
    nb_ghosts: int

    meshio_cells: npt.NDArray[np.int32]
    cells: npt.NDArray[np.int32]
    nodes: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int32]
    cell_faceid: npt.NDArray[np.int32]
    phy_faces: npt.NDArray[np.int32]
    face_to_phyid: npt.NDArray[np.int32]
    phy_id_to_face_id: npt.NDArray[np.int32]
    face_cellid: npt.NDArray[np.int32]
    cell_cellnid: npt.NDArray[np.int32]
    cell_cellfid: npt.NDArray[np.int32]
    node_cellid: npt.NDArray[np.int32]
    cell_ghostnid: npt.NDArray[np.int32]
    node_ghostid: npt.NDArray[np.int32]
    face_oldname: npt.NDArray[np.int32]
    node_oldname: npt.NDArray[np.int32]

    cell_center: npt.NDArray[np.float64]
    cell_volume: npt.NDArray[np.float64]
    face_center: npt.NDArray[np.float64]
    face_normal: npt.NDArray[np.float64]
    face_measure: npt.NDArray[np.float64]
    ghost_info_flt: npt.NDArray[np.float64]
    ghost_info_int: npt.NDArray[np.int32]
