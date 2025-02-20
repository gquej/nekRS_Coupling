#ifndef COUPLING_HPP
#define COUPLING_HPP


//#include "nrs.hpp"

#include <iostream>
#include <algorithm> // for std::copy
#include <occa.hpp>
#include "mpi.h"
#include <vector>
#include <precice/precice.hpp>
using vector_t = std::vector<double>;
using std::vector;
using string_t = precice::string_view;


class Coupling {
    protected:
        double * vertices_coord_;
        int * vertices_mapping_;
        double * data_;
        double foo_;
        int mesh_size_;

        occa::memory o_data;
        occa::memory o_mapping;


        //precice stuff
        precice::Participant *precice_;
        std::string_view solver_name_; //!< The name of this solver
        std::string_view config_file_; //!< The path to the preCICE configuration file


        //stuff of this mesh, and that must be received
        vector<int> mapping_;
        precice::string_view mesh_name_;
        vector_t vertices_; //The vertices, stored as [x1,y1,z1,x2,y2,z2,..., xn,yn,zn]
        vector<int> vertex_IDs_; //The vertices IDs
        string_t data_name_;
        vector_t Data_; //The data to be sent or received



        //other mesh (direct ) stuff, and stuff that must be sent to murphy 

        vector_t bounding_box_;
        int direct_mesh_size_;
        precice::string_view direct_mesh_name_;
        vector_t direct_vertices_;
        vector<int> direct_vertex_IDs_;
        string_t direct_data_name_;
        string_t direct_participant_name_;
        vector_t direct_data_; //The data to be sent or received


    public: 
        Coupling() = default;
        explicit Coupling (std::string_view solver_name, std::string_view config_file);
        void Setup(precice::string_view mesh_name, precice::string_view direct_mesh_name, precice::string_view data_name, double *bounding_box);
        void Allocate_mapping(int mapping_size);
        void Allocate_vertices(double * vertices, int size);
        void Set_Vertices_coord(double * vertices) ;
        double * Get_Vertices_coord() {return vertices_coord_;};
        double * Get_data() {return data_;};
        int * Get_Vertices_mapping() {return vertices_mapping_;};

        occa::memory Get_o_data() {return o_data;};
        occa::memory Get_o_mapping() {return o_mapping;};

        void Set_o_data(occa::memory temp) {o_data = temp;};
        void Set_o_mapping(occa::memory temp) {o_mapping = temp;};

        void Read(double dt);
        void Write();
        void Advance(double dt) {precice_->advance(dt);};
        void Finalize() {precice_->finalize();};
        

        ~Coupling() {
            delete[] vertices_coord_;
            delete[] vertices_mapping_;
            delete[] data_;
        }



};


#endif