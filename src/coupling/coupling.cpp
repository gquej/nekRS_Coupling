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

}


