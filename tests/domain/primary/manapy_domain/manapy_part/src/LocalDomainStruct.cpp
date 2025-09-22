#define PY_ARRAY_UNIQUE_SYMBOL MYPACKAGE_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "LocalDomainStruct.h"
#include <algorithm>

LocalDomainStruct::~LocalDomainStruct() {
    this->free_tables();
    Py_XDECREF(this->tuple_res);
    this->tuple_res = nullptr;
}

void LocalDomainStruct::_create_tables(PyArray<double, 2> *nodes, std::vector<int32_t> &part_phyid) {
    // #########################################################
    // vec_phyids, map_phyids, nb_halos_int
    // #########################################################

    const auto &set_phyids = this->set_phyids;
    this->map_phyids = std::map<int32_t, int32_t>();
    this->vec_phyids = std::vector<int32_t>(set_phyids.size());

    auto &vec_phyids = this->vec_phyids;
    auto &map_phyids = this->map_phyids;
    int32_t nb_halos_int = 0;

    int32_t counter = 0;
    for (const auto item : set_phyids) {
        vec_phyids[counter] = item;
        counter += 1;
    }
    // Sort vec_phyids by comparing part_phyid[phyid]
    std::sort(vec_phyids.begin(), vec_phyids.end(),[&part_phyid](const int a, const int b) {
        return part_phyid[a] < part_phyid[b];
    });

    for (int32_t i = 0; i < vec_phyids.size(); i++) {
        const int32_t item = vec_phyids[i];
        map_phyids[item] = i;
    }

    for (const auto &item : this->map_halo_int) {
        nb_halos_int += (int32_t)this->map_halo_int[item.first].size();
    }
    // #########################################################
    // Tables
    // #########################################################

    // this->map_phyids = map_phyids; assigned above
    // this->vec_phyids = vec_phyids; assigned above
    this->nodes = new PyArray<double, 2>(make_npy_dims(this->map_nodes.size(), nodes->shape[1]));
    this->cells = new PyArray<int32_t, 2>(make_npy_dims(this->map_cells.size(), this->max_cell_nodeid + 1));
    this->cells_type = new PyArray<int8_t, 1>(make_npy_dims(this->map_cells.size()));
    this->phy_faces = new PyArray<int32_t, 2>(make_npy_dims(this->map_phy_faces.size(), this->max_phy_face_nodeid + 1));
    this->phy_faces_name = new PyArray<int32_t, 1>(make_npy_dims(this->map_phy_faces.size()));
    this->cell_loctoglob = new PyArray<int32_t, 1>(make_npy_dims(this->map_cells.size()));
    this->node_loctoglob = new PyArray<int32_t, 1>(make_npy_dims(this->map_nodes.size()));
    this->node_oldname = new PyArray<int32_t, 1>(make_npy_dims(this->map_nodes.size()));
    this->halo_neighsub = new PyArray<int32_t, 2>(make_npy_dims(2, this->map_halo_int.size()));
    this->node_halos = new PyArray<int32_t, 1>(make_npy_dims(this->nb_node_halos));
    this->node_halophyid = new PyArray<int32_t, 2>(make_npy_dims(this->map_nodes.size(), this->max_node_halophyid + 1));
    this->phyid_recv = new PyArray<int32_t, 1>(make_npy_dims(this->vec_phyids.size()));
    this->phyid_recv_part_size = new PyArray<int32_t, 1>(make_npy_dims(this->set_halo_phyid_neighsub.size() * 2 + 2));
    // this->phyid_send -> created at _create_phyid_send
    this->halo_halosext = new PyArray<int32_t, 2>(make_npy_dims(this->map_halos.size(), this->max_cell_nodeid + 2));
    this->halo_halosint = new PyArray<int32_t, 1>(make_npy_dims(nb_halos_int));
}

void LocalDomainStruct::create_tuple() {
    PyObject *tuple = Py_BuildValue("(OOOOOOOOOOOOOOOOiiiii)",
        this->nodes->ref_holder,
        this->cells->ref_holder,
        this->cells_type->ref_holder,
        this->phy_faces->ref_holder,
        this->phy_faces_name->ref_holder,
        this->cell_loctoglob->ref_holder,
        this->node_loctoglob->ref_holder,
        this->node_oldname->ref_holder,
        this->halo_neighsub->ref_holder,
        this->node_halos->ref_holder,
        this->node_halophyid->ref_holder,
        this->phyid_recv->ref_holder,
        this->phyid_recv_part_size->ref_holder,
        this->phyid_send->ref_holder,
        this->halo_halosext->ref_holder,
        this->halo_halosint->ref_holder,
        this->max_cell_nodeid,
        this->max_cell_faceid,
        this->max_face_nodeid,
        this->max_node_haloid,
        this->max_cell_halonid);
    if (!tuple) {
        throw std::bad_alloc();
    }

    // tuple hold references now
    this->free_tables();
    this->tuple_res = tuple;
}


// private
void LocalDomainStruct::free_tables() {
    // all these tables are created using this->_create_tables
    delete this->nodes; this->nodes = nullptr;
    delete this->cells; this->cells = nullptr;
    delete this->cells_type; this->cells_type = nullptr;
    delete this->phy_faces; this->phy_faces = nullptr;
    delete this->phy_faces_name; this->phy_faces_name = nullptr;
    delete this->cell_loctoglob; this->cell_loctoglob = nullptr;
    delete this->node_loctoglob; this->node_loctoglob = nullptr;
    delete this->node_oldname; this->node_oldname = nullptr;
    delete this->halo_neighsub; this->halo_neighsub = nullptr;
    delete this->node_halos; this->node_halos = nullptr;
    delete this->node_halophyid; this->node_halophyid = nullptr;
    delete this->phyid_recv; this->phyid_recv = nullptr;
    delete this->phyid_recv_part_size; this->phyid_recv_part_size = nullptr;
    delete this->phyid_send; this->phyid_send = nullptr;
    delete this->halo_halosext; this->halo_halosext = nullptr;
    delete this->halo_halosint; this->halo_halosint = nullptr;
}