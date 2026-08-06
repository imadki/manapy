#include "domain_compute.hpp"

#include <stdexcept>

void variables_3d(ArrayView<const real_t, 2> cell_center,
                   ArrayView<const index_t, 2> node_cellid,
                   ArrayView<const index_t, 2> node_haloid,
                   ArrayView<const index_t, 2> node_ghostid,
                   ArrayView<const index_t, 2> node_haloghostid,
                   ArrayView<const index_t, 2> node_periodicid,
                   ArrayView<const real_t, 2> nodes,
                   ArrayView<const index_t, 1> node_oldname,
                   ArrayView<const real_t, 2> ghost_info_flt,
                   ArrayView<const real_t, 2> ext_ghost_info_flt,
                   ArrayView<const real_t, 2> halo_centvol,
                   ArrayView<real_t, 1> node_R_x, ArrayView<real_t, 1> node_R_y,
                   ArrayView<real_t, 1> node_R_z,
                   ArrayView<real_t, 1> node_lambda_x,
                   ArrayView<real_t, 1> node_lambda_y,
                   ArrayView<real_t, 1> node_lambda_z,
                   ArrayView<index_t, 1> node_number,
                   ArrayView<const real_t, 2> cell_shift) {
  const index_t nbnode = static_cast<index_t>(node_R_x.size(0));

  for (index_t i = 0; i < nbnode; ++i) {
    real_t i_xx = real_t(0), i_yy = real_t(0), i_zz = real_t(0);
    real_t i_xy = real_t(0), i_xz = real_t(0), i_yz = real_t(0);
    const real_t nx = nodes(i, 0);
    const real_t ny = nodes(i, 1);
    const real_t nz = nodes(i, 2);

    auto accumulate = [&](real_t cx, real_t cy, real_t cz) {
      const real_t rx = cx - nx;
      const real_t ry = cy - ny;
      const real_t rz = cz - nz;
      i_xx += rx * rx;
      i_yy += ry * ry;
      i_zz += rz * rz;
      i_xy += rx * ry;
      i_xz += rx * rz;
      i_yz += ry * rz;
      node_R_x(i) += rx;
      node_R_y(i) += ry;
      node_R_z(i) += rz;
      node_number(i) += 1;
    };

    const auto cellid_row = node_cellid.row(i);
    const index_t cellid_count = cellid_row(cellid_row.size(0) - 1);
    for (index_t j = 0; j < cellid_count; ++j) {
      const index_t cell = cellid_row(j);
      accumulate(cell_center(cell, 0), cell_center(cell, 1),
                 cell_center(cell, 2));
    }

    const auto ghostid_row = node_ghostid.row(i);
    const index_t ghostid_count = ghostid_row(ghostid_row.size(0) - 1);
    for (index_t j = 0; j < ghostid_count; ++j) {
      const index_t g = ghostid_row(j);
      accumulate(ghost_info_flt(g, 0), ghost_info_flt(g, 1),
                 ghost_info_flt(g, 2));
    }

    // Periodic boundary (old vertex names). One unified branch: apply the
    // FULL cell_shift vector so a partner cell coming from ANY periodic
    // direction is imaged correctly. An edge/corner node carries partners
    // from more than one direction (each partner cell holds its own
    // already-signed shift, zero on the components it's not periodic in).
    const index_t oldname = node_oldname(i);
    if (oldname >= 11) {
      const auto periodicid_row = node_periodicid.row(i);
      const index_t count = periodicid_row(periodicid_row.size(0) - 1);
      for (index_t j = 0; j < count; ++j) {
        const index_t cell = periodicid_row(j);
        accumulate(cell_center(cell, 0) + cell_shift(cell, 0),
                   cell_center(cell, 1) + cell_shift(cell, 1),
                   cell_center(cell, 2) + cell_shift(cell, 2));
      }
    }

    const auto haloid_row = node_haloid.row(i);
    const index_t haloid_count = haloid_row(haloid_row.size(0) - 1);
    for (index_t j = 0; j < haloid_count; ++j) {
      const index_t cell = haloid_row(j);
      accumulate(halo_centvol(cell, 0), halo_centvol(cell, 1),
                 halo_centvol(cell, 2));
    }

    const auto haloghostid_row = node_haloghostid.row(i);
    const index_t haloghostid_count =
        haloghostid_row(haloghostid_row.size(0) - 1);
    for (index_t j = 0; j < haloghostid_count; ++j) {
      const index_t cell = haloghostid_row(j);
      accumulate(ext_ghost_info_flt(cell, 0), ext_ghost_info_flt(cell, 1),
                 ext_ghost_info_flt(cell, 2));
    }

    const real_t d = i_xx * i_yy * i_zz + real_t(2) * i_xy * i_xz * i_yz -
                      i_xx * i_yz * i_yz - i_yy * i_xz * i_xz -
                      i_zz * i_xy * i_xy;
    if (d == real_t(0))
      throw std::runtime_error("variables_3d: div 0");

    const real_t rx = node_R_x(i), ry = node_R_y(i), rz = node_R_z(i);

    node_lambda_x(i) = ((i_yz * i_yz - i_yy * i_zz) * rx +
                        (i_xy * i_zz - i_xz * i_yz) * ry +
                        (i_xz * i_yy - i_xy * i_yz) * rz) /
                       d;
    node_lambda_y(i) = ((i_xy * i_zz - i_xz * i_yz) * rx +
                        (i_xz * i_xz - i_xx * i_zz) * ry +
                        (i_yz * i_xx - i_xz * i_xy) * rz) /
                       d;
    node_lambda_z(i) = ((i_xz * i_yy - i_xy * i_yz) * rx +
                        (i_yz * i_xx - i_xz * i_xy) * ry +
                        (i_xy * i_xy - i_xx * i_yy) * rz) /
                       d;
  }
}
