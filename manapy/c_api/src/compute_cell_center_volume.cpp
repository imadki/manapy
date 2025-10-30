#ifndef COMPUTE_CELL_CENTER_VOLUME_H
#define COMPUTE_CELL_CENTER_VOLUME_H

#include "manapy_part.h"

void compute_halo_cell_center_area_2d(PyArray<int32_t, 2> const *halo_halosext, PyArray<fdx_t, 2> const *nodes, PyArray<fdx_t, 2> *halo_centvol) {
    // ** This code is the same as compute_cell_center_area_2d any change to this function may imply also changing the latter function ** //
    //Area using shoelace formula (also called Gauss’s area formula or the surveyors' formula).
    //Area = 1/2 * | Σ (x_i * y_{i+1} − x_{i+1} * y_i)

    double p[4][2]; //2D square, triangle

    for (int32_t i = 0; i <halo_halosext->shape[0]; i++) {
        double area = 0.0;
        const int32_t nb_vertex = halo_halosext->last2(i) - 1;  //skipping cell_id in halo_halosext[i]

        // copy vertices
        for (int32_t j = 0; j < nb_vertex; j++) {
            const int32_t node_id = halo_halosext->get2(i, j + 1); //skipping cell_id in halo_halosext[i]
            p[j][0] = nodes->get2(node_id, 0);
            p[j][1] = nodes->get2(node_id, 1);
        }

        if (nb_vertex == 3) { // triangle
            //## Center
            halo_centvol->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0]) / 3.0);
            halo_centvol->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1]) / 3.0);
            halo_centvol->get2(i, 2) = 0.0; // ??

            //#Area (polygon_area_2d)
            area += p[0][0] * p[1][1] - p[1][0] * p[0][1];
            area += p[1][0] * p[2][1] - p[2][0] * p[1][1];
            area += p[2][0] * p[0][1] - p[0][0] * p[2][1];
            area = std::abs(area) / 2.0;
        } else if (nb_vertex == 4) { // square
            //## Center
            halo_centvol->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4.0);
            halo_centvol->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0);
            halo_centvol->get2(i, 2) = 0.0; // ??

            //#Area
            area += p[0][0] * p[1][1] - p[1][0] * p[0][1];
            area += p[1][0] * p[2][1] - p[2][0] * p[1][1];
            area += p[2][0] * p[3][1] - p[3][0] * p[2][1];
            area += p[3][0] * p[0][1] - p[0][0] * p[3][1];
            area = std::abs(area) / 2.0;
        }
        halo_centvol->get2(i, 3) = static_cast<fdx_t>(area);
    }
}

void compute_halo_cell_center_volume_3d(PyArray<int32_t, 2> const *halo_halosext, PyArray<fdx_t, 2> const *nodes, PyArray<fdx_t, 2> *halo_centvol) {
    // ** This code is the same as compute_cell_center_volume_3d any change to this function may imply also changing the latter function ** //
    double p[8][3]; //3D Tetrahedron Hexahedron Pyramid

    const auto _tetrahedron_volume = [](const double *a, const double *b, const double *c, const double *d) {
        // compute det[b - a, c - a, d - a] / 6
        return ((b[0]-a[0])*((c[1]-a[1])*(d[2]-a[2]) - (c[2]-a[2])*(d[1]-a[1]))
              + (b[1]-a[1])*((c[2]-a[2])*(d[0]-a[0]) - (c[0]-a[0])*(d[2]-a[2]))
              + (b[2]-a[2])*((c[0]-a[0])*(d[1]-a[1]) - (c[1]-a[1])*(d[0]-a[0])))/6.0;
    };

    for (int32_t i = 0; i <halo_halosext->shape[0]; i++) {
        const int32_t nb_vertex = halo_halosext->last2(i) - 1; //skipping cell_id in halo_halosext[i]

        // copy vertices
        for (int32_t j = 0; j < nb_vertex; j++) {
            const int32_t node_id = halo_halosext->get2(i, j + 1); //skipping cell_id in halo_halosext[i]
            p[j][0] = nodes->get2(node_id, 0);
            p[j][1] = nodes->get2(node_id, 1);
            p[j][2] = nodes->get2(node_id, 2);
        }

        //Calculate Center and Volume
        double vol = 0.0;
        if (nb_vertex == 4) { // Tetrahedron
            //## Center
            halo_centvol->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4.0);
            halo_centvol->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0);
            halo_centvol->get2(i, 2) = static_cast<fdx_t>((p[0][2] + p[1][2] + p[2][2] + p[3][2]) / 4.0);

            //## Volume
            vol += _tetrahedron_volume(p[0], p[1], p[2], p[3]);
        } else if (nb_vertex == 8) { // Hexahedron
            //## Center
            halo_centvol->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0] + p[3][0] + p[4][0] + p[5][0] + p[6][0] + p[7][0]) / 8.0);
            halo_centvol->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1] + p[3][1] + p[4][1] + p[5][1] + p[6][1] + p[7][1]) / 8.0);
            halo_centvol->get2(i, 2) = static_cast<fdx_t>((p[0][2] + p[1][2] + p[2][2] + p[3][2] + p[4][2] + p[5][2] + p[6][2] + p[7][2]) / 8.0);

            //## Volume
            // [0, 1, 3, 4], # 1 tetra
            // [1, 3, 4, 5], # 2 tetra
            // [4, 5, 3, 7], # 3 tetra
            // [1, 3, 5, 2], # 4 tetra
            // [3, 7, 5, 2], # 5 tetra
            // [5, 7, 6, 2]  # 6 tetra
            vol += _tetrahedron_volume(p[0], p[1], p[3], p[4]);
            vol += _tetrahedron_volume(p[1], p[3], p[4], p[5]);
            vol += _tetrahedron_volume(p[4], p[5], p[3], p[7]);
            vol += _tetrahedron_volume(p[1], p[3], p[5], p[2]);
            vol += _tetrahedron_volume(p[3], p[7], p[5], p[2]);
            vol += _tetrahedron_volume(p[5], p[7], p[6], p[2]);
        } else if (nb_vertex == 5) { // Pyramid
            halo_centvol->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0] + p[3][0] + p[4][0]) / 5.0);
            halo_centvol->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1] + p[3][1] + p[4][1]) / 5.0);
            halo_centvol->get2(i, 2) = static_cast<fdx_t>((p[0][2] + p[1][2] + p[2][2] + p[3][2] + p[4][2]) / 5.0);
            // [0, 1, 2, 4],  # 1 tetra
            // [0, 2, 3, 4],  # 2 tetra
            vol += _tetrahedron_volume(p[0], p[1], p[2], p[4]);
            vol += _tetrahedron_volume(p[0], p[2], p[3], p[4]);
        }
        halo_centvol->get2(i, 3) = static_cast<fdx_t>(vol);
    }
}

void compute_cell_center_area_2d(PyArray<int32_t, 2> const *cells, PyArray<fdx_t, 2> const *nodes, PyArray<fdx_t, 1> *cell_area, PyArray<fdx_t, 2> *cell_center) {
    // ** This code is the same as compute_halo_cell_center_volume_3d any change to this function may imply also changing the latter function ** //
    //Area using shoelace formula (also called Gauss’s area formula or the surveyors' formula).
    //Area = 1/2 * | Σ (x_i * y_{i+1} − x_{i+1} * y_i)

    double p[4][2]; //2D square, triangle

    for (int32_t i = 0; i <cells->shape[0]; i++) {
        double area = 0.0;
        const int32_t nb_vertex = cells->last2(i);

        // copy vertices
        for (int32_t j = 0; j < nb_vertex; j++) {
            const int32_t node_id = cells->get2(i, j);
            p[j][0] = nodes->get2(node_id, 0);
            p[j][1] = nodes->get2(node_id, 1);
        }

        if (nb_vertex == 3) { // triangle
            //## Center
            cell_center->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0]) / 3.0);
            cell_center->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1]) / 3.0);

            //#Area (polygon_area_2d)
            area += p[0][0] * p[1][1] - p[1][0] * p[0][1];
            area += p[1][0] * p[2][1] - p[2][0] * p[1][1];
            area += p[2][0] * p[0][1] - p[0][0] * p[2][1];
            area = std::abs(area) / 2.0;
        } else if (nb_vertex == 4) { // square
            //## Center
            cell_center->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4.0);
            cell_center->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0);

            //#Area
            area += p[0][0] * p[1][1] - p[1][0] * p[0][1];
            area += p[1][0] * p[2][1] - p[2][0] * p[1][1];
            area += p[2][0] * p[3][1] - p[3][0] * p[2][1];
            area += p[3][0] * p[0][1] - p[0][0] * p[3][1];
            area = std::abs(area) / 2.0;
        }
        cell_area->get(i) = static_cast<fdx_t>(area);
    }
}

void compute_cell_center_volume_3d(PyArray<int32_t, 2> const *cells, PyArray<fdx_t, 2> const *nodes, PyArray<fdx_t, 1> *cell_volume, PyArray<fdx_t, 2> *cell_center) {
    // ** This code is the same as compute_halo_cell_center_volume_3d any change to this function may imply also changing the latter function ** //
    double p[8][3]; //3D Tetrahedron Hexahedron Pyramid

    const auto _tetrahedron_volume = [](const double *a, const double *b, const double *c, const double *d) {
        // compute det[b - a, c - a, d - a] / 6
        return ((b[0]-a[0])*((c[1]-a[1])*(d[2]-a[2]) - (c[2]-a[2])*(d[1]-a[1]))
              + (b[1]-a[1])*((c[2]-a[2])*(d[0]-a[0]) - (c[0]-a[0])*(d[2]-a[2]))
              + (b[2]-a[2])*((c[0]-a[0])*(d[1]-a[1]) - (c[1]-a[1])*(d[0]-a[0])))/6.0;
    };

    for (int32_t i = 0; i <cells->shape[0]; i++) {
        const int32_t nb_vertex = cells->last2(i);

        // copy vertices
        for (int32_t j = 0; j < nb_vertex; j++) {
            const int32_t node_id = cells->get2(i, j);
            p[j][0] = nodes->get2(node_id, 0);
            p[j][1] = nodes->get2(node_id, 1);
            p[j][2] = nodes->get2(node_id, 2);
        }

        //Calculate Center and Volume
        double vol = 0.0;
        if (nb_vertex == 4) { // Tetrahedron
            //## Center
            cell_center->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4.0);
            cell_center->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0);
            cell_center->get2(i, 2) = static_cast<fdx_t>((p[0][2] + p[1][2] + p[2][2] + p[3][2]) / 4.0);

            //## Volume
            vol += _tetrahedron_volume(p[0], p[1], p[2], p[3]);
        } else if (nb_vertex == 8) { // Hexahedron
            //## Center
            cell_center->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0] + p[3][0] + p[4][0] + p[5][0] + p[6][0] + p[7][0]) / 8.0);
            cell_center->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1] + p[3][1] + p[4][1] + p[5][1] + p[6][1] + p[7][1]) / 8.0);
            cell_center->get2(i, 2) = static_cast<fdx_t>((p[0][2] + p[1][2] + p[2][2] + p[3][2] + p[4][2] + p[5][2] + p[6][2] + p[7][2]) / 8.0);

            //## Volume
            // [0, 1, 3, 4], # 1 tetra
            // [1, 3, 4, 5], # 2 tetra
            // [4, 5, 3, 7], # 3 tetra
            // [1, 3, 5, 2], # 4 tetra
            // [3, 7, 5, 2], # 5 tetra
            // [5, 7, 6, 2]  # 6 tetra
            vol += _tetrahedron_volume(p[0], p[1], p[3], p[4]);
            vol += _tetrahedron_volume(p[1], p[3], p[4], p[5]);
            vol += _tetrahedron_volume(p[4], p[5], p[3], p[7]);
            vol += _tetrahedron_volume(p[1], p[3], p[5], p[2]);
            vol += _tetrahedron_volume(p[3], p[7], p[5], p[2]);
            vol += _tetrahedron_volume(p[5], p[7], p[6], p[2]);
        } else if (nb_vertex == 5) { // Pyramid
            cell_center->get2(i, 0) = static_cast<fdx_t>((p[0][0] + p[1][0] + p[2][0] + p[3][0] + p[4][0]) / 5.0);
            cell_center->get2(i, 1) = static_cast<fdx_t>((p[0][1] + p[1][1] + p[2][1] + p[3][1] + p[4][1]) / 5.0);
            cell_center->get2(i, 2) = static_cast<fdx_t>((p[0][2] + p[1][2] + p[2][2] + p[3][2] + p[4][2]) / 5.0);
            // [0, 1, 2, 4],  # 1 tetra
            // [0, 2, 3, 4],  # 2 tetra
            vol += _tetrahedron_volume(p[0], p[1], p[2], p[4]);
            vol += _tetrahedron_volume(p[0], p[2], p[3], p[4]);
        }
        cell_volume->get(i) = static_cast<fdx_t>(vol);
    }
}

#endif //COMPUTE_CELL_CENTER_VOLUME_H
