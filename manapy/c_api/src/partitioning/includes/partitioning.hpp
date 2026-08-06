#pragma once

// The partitioning pipeline's public entry point. In the c_api this lived in
// manapy_part.h, which included LocalDomainStruct.h to get the struct; here the
// two are split, so the declaration that needs both sits in its own header.

#include "local_domain_struct.hpp"

/**
 * @brief Top-level entry point: splits the global mesh into @p nb_parts local subdomains.
 *
 * See src/partitioning.cpp for the pipeline description. `ld` must point at
 * nb_parts default-constructed LocalDomainStruct; on return each has been
 * emptied into the corresponding tuple of the returned list.
 */
nb::list create_sub_domains(LocalDomainStruct *ld,
                            ArrayView<const index_t, 1> part_vert,
                            ArrayView<const index_t, 2> node_cellid,
                            ArrayView<const real_t, 2> nodes,
                            ArrayView<const index_t, 2> cells,
                            ArrayView<const int8_t, 1> cells_type,
                            ArrayView<const index_t, 2> phy_faces,
                            ArrayView<const index_t, 1> phy_faces_name,
                            ArrayView<const index_t, 2> node_phyid,
                            index_t nb_parts, index_t dim);
