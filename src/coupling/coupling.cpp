#include "coupling/coupling.hpp"

void Coupling::Set_Vertices_coord(double * vertices) {
    vertices_coord_ = vertices;
}

void Coupling::Allocate_mapping(int mapping_size) {
    vertices_mapping_ = new int[mapping_size];
}

void Coupling::Allocate_vertices(double * vertices, int size) {
    vertices_coord_ = new double[3 * size];
    std::copy(vertices, vertices + 3 * size, vertices_coord_);

    data_ = new double[3 * size];
    mesh_size_ = size;

}

Coupling::Coupling(std::string_view solver_name, std::string_view config_file) 
: solver_name_(solver_name), config_file_(config_file){

    int rank;
    int comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    precice_ = new precice::Participant(solver_name, config_file, rank, comm_size);

}

void Coupling::Setup(precice::string_view mesh_name, precice::string_view direct_mesh_name, precice::string_view data_name, double *bounding_box) {
    mesh_name_ = mesh_name;
    direct_mesh_name_ = direct_mesh_name;
    data_name_ = data_name;
    bounding_box_.resize(6);
    for (int i = 0; i < 6; i++)
    {
        bounding_box_[i] = bounding_box[i];
    }
    
    //setup this mesh
    vertices_.resize(3*mesh_size_);
    vertex_IDs_.resize(mesh_size_);
    for (int i = 0; i < mesh_size_; i++)
    {
        vertices_[3*i] = vertices_coord_[3*i];
        vertices_[3*i + 1] = vertices_coord_[3*i + 1];
        vertices_[3*i + 2] = vertices_coord_[3*i + 2];
    }
    
    precice_->setMeshVertices(mesh_name_, vertices_, vertex_IDs_);

    //setup the direct mesh 
    precice_->setMeshAccessRegion(direct_mesh_name_, bounding_box_);
    precice_->initialize();
    //finalize direct mesh setup
    direct_mesh_size_ = precice_->getMeshVertexSize(direct_mesh_name_);
    direct_vertices_.resize(3 * direct_mesh_size_);
    direct_data_.resize(3 * direct_mesh_size_);
    direct_vertex_IDs_.resize(direct_mesh_size_);
    precice_->getMeshVertexIDsAndCoordinates(direct_mesh_name_, direct_vertex_IDs_, direct_vertices_);

}

void Coupling::Read(double dt) {
    precice_->readData(mesh_name_, data_name_, vertex_IDs_, dt, Data_);
}

void Coupling::Write() {
    for (int i = 0; i < direct_mesh_size_; i++)
    {
        direct_data_[i] = 0.;
    }
    
    precice_->writeData(direct_mesh_name_, direct_data_name_, direct_vertex_IDs_, direct_data_);
}
