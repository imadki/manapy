#include "domain_compute.hpp"

#include <cmath>

void create_ghost_info(ArrayView<const index_t, 2> bf_cellid,
                        ArrayView<const real_t, 2> cell_center,
                        ArrayView<const index_t, 2> cell_faceid,
                        ArrayView<const index_t, 1> cell_loctoglob,
                        ArrayView<const index_t, 2> faces,
                        ArrayView<const real_t, 2> nodes,
                        ArrayView<const index_t, 1> face_oldname,
                        ArrayView<const real_t, 2> face_normal,
                        ArrayView<const real_t, 2> face_center,
                        ArrayView<const real_t, 1> face_measure,
                        ArrayView<index_t, 2> ghost_info_int,
                        ArrayView<real_t, 2> ghost_info_flt, index_t dim) {
  const index_t nb = static_cast<index_t>(bf_cellid.size(0));
  const bool has_loctoglob = cell_loctoglob.size(0) != 0;

  for (index_t i = 0; i < nb; ++i) {
    const index_t cid = bf_cellid(i, 0);
    if (cid == -1) {
      // Periodic face: mark invalid so ghost tables skip it.
      ghost_info_int(i, 0) = -1;
      continue;
    }
    const index_t bf = bf_cellid(i, 1); // face index inside the cell
    const index_t fid = cell_faceid(cid, bf);

    const real_t fcx = face_center(fid, 0), fcy = face_center(fid, 1),
                 fcz = face_center(fid, 2);
    const real_t ccx = cell_center(cid, 0), ccy = cell_center(cid, 1),
                 ccz = cell_center(cid, 2);
    const real_t fnx = face_normal(fid, 0), fny = face_normal(fid, 1),
                 fnz = face_normal(fid, 2);

    const real_t norm = std::sqrt(fnx * fnx + fny * fny + fnz * fnz);
    const real_t nhx = fnx / norm, nhy = fny / norm, nhz = fnz / norm;

    const real_t dx = ccx - fcx, dy = ccy - fcy, dz = ccz - fcz;
    const real_t proj = dx * nhx + dy * nhy + dz * nhz;

    const real_t gcx = ccx - real_t(2) * proj * nhx;
    const real_t gcy = ccy - real_t(2) * proj * nhy;
    const real_t gcz = ccz - real_t(2) * proj * nhz;

    real_t gamma;
    if (dim == 2) {
      const index_t n0 = faces(fid, 0);
      const index_t n1 = faces(fid, 1);
      const real_t p0x = nodes(n0, 0), p0y = nodes(n0, 1), p0z = nodes(n0, 2);
      const real_t p1x = nodes(n1, 0), p1y = nodes(n1, 1), p1z = nodes(n1, 2);

      const real_t ux = ccx - p1x, uy = ccy - p1y, uz = ccz - p1z;
      const real_t vx = p0x - p1x, vy = p0y - p1y, vz = p0z - p1z;
      const real_t measure = face_measure(fid);
      gamma = (ux * vx + uy * vy + uz * vz) / (measure * measure);
    } else {
      const real_t ux = fcx - ccx, uy = fcy - ccy, uz = fcz - ccz;
      const real_t inv_measure = real_t(1) / face_measure(fid);
      const real_t nx = fnx * inv_measure, ny = fny * inv_measure,
                   nz = fnz * inv_measure;
      gamma = ux * nx + uy * ny + uz * nz;
    }

    ghost_info_flt(i, 0) = gcx;
    ghost_info_flt(i, 1) = gcy;
    ghost_info_flt(i, 2) = gcz;
    ghost_info_flt(i, 3) = gamma;
    ghost_info_flt(i, 4) = fcx;
    ghost_info_flt(i, 5) = fcy;
    ghost_info_flt(i, 6) = fcz;
    ghost_info_flt(i, 7) = fnx;
    ghost_info_flt(i, 8) = fny;
    ghost_info_flt(i, 9) = fnz;

    ghost_info_int(i, 0) = cid;
    ghost_info_int(i, 1) = bf;
    ghost_info_int(i, 2) = face_oldname(fid);
    if (has_loctoglob)
      ghost_info_int(i, 3) = cell_loctoglob(cid);
    ghost_info_int(i, 4) = fid;
  }
}
