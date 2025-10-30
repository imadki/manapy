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

void LocalDomainStruct::create_tuple() {
    PyObject *tuple = Py_BuildValue("(OOOOOOOOOOOOOOOOOiiiii)",
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
        this->halo_centvol->ref_holder,
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
    delete this->halo_centvol; this->halo_centvol = nullptr;
}