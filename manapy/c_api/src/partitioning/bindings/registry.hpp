#pragma once

#include "manapy_compute_types.hpp"

// Each partitioning binding TU exposes one register_* function that adds its
// Python entry points to the module; bindings/module.cpp defines
// NB_MODULE(_core) and calls them all.

// metis_partitioning.cpp: the three METIS wrappers -- make_n_part_graph_k_way,
// make_n_part_mesh_dual, make_n_part_mesh_nodal.
void register_metis_partitioning(nb::module_ &m);

// create_local_domains.cpp: the partitioning pipeline itself.
void register_create_local_domains(nb::module_ &m);
