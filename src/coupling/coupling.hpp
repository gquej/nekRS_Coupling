#ifndef COUPLING_HPP
#define COUPLING_HPP


//#include "nrs.hpp"

#include <iostream>
#include <algorithm> // for std::copy
#include "mpi.h"
#include <vector>
#include <precice/precice.hpp>
using vector_t = std::vector<double>;
using std::vector;
using string_t = precice::string_view;


class Coupling {
    protected:
        precice::Participant *precice_;
        string_t solver_name_; //!< The name of this solver
        string_t config_file_; //!< The path to the preCICE configuration file

        //this mesh and receive -----------------------------
        string_t mesh_name_; //name of this mesh
        int mesh_size_; //size of this mesh
        vector_t vertices_; //The vertices, stored as [x1,y1,z1,x2,y2,z2,..., xn,yn,zn]
        vector<int> vertex_IDs_; //The vertices IDs
        vector<int> mapping_; //Mapping from nek to the precice buffer

        string_t data1_name_; //data 1: murphy velocity
        vector_t Data1_; 

        string_t data2_name_; //data 2: murphy vorticity
        vector_t Data2_;

        //directly accessed mesh and send -----------------------------
        string_t direct_mesh_name_;
        string_t direct_participant_name_;
        int direct_mesh_size_;
        vector_t bounding_box_;
        vector_t direct_vertices_;
        vector<int> direct_vertex_IDs_;

        string_t direct_data_name_; //The data to be sent or received
        vector_t direct_data_; 

        string_t direct_data_name2_;  // The Nek mesh coordinates
        vector_t direct_data2_;
        string_t direct_data_name3_;  // The Nek mesh velocity
        vector_t direct_data3_;
        vector_t global_data_; // Global mesh position
        vector_t global_data2_; // Global mesh orientation

        string_t direct_data_name_cum_; //The cumulative data to be sent or received
        vector_t direct_data_cum_; 


    public: 
        Coupling() = default;
        explicit Coupling (std::string_view solver_name, std::string_view config_file);
        void Setup(precice::string_view mesh_name, precice::string_view direct_mesh_name, precice::string_view data_name, double *bounding_box, precice::string_view data2_name, precice::string_view direct_data_name, precice::string_view direct_data_name2, precice::string_view direct_data_name3, precice::string_view direct_data_name_cum);
        void Resize_mapping(int mapping_size) {mapping_.resize(mapping_size);};
        void Set_vertices(double *vertices, int size);
        void Resetup();

        void Read(double dt);
        void Write();
        void Advance(double dt) {precice_->advance(dt);};
        void Finalize() {precice_->finalize(); delete precice_;};
        double GetMaxTimeStep() {return precice_->getMaxTimeStepSize();};

        double * Get_data1() {return Data1_.data();};
        double * Get_data2() {return Data2_.data();};
        int * Get_mapping() {return mapping_.data();};
        vector_t *direct_vertices() {return &direct_vertices_; };
        int mesh_size() {return mesh_size_;};
        int direct_mesh_size() {return direct_mesh_size_;};
        vector_t *direct_data() {return &direct_data_;};
        vector_t *direct_data_cum() {return &direct_data_cum_;};
        vector_t *direct_data2() {return &direct_data2_;};
        vector_t *direct_data3() {return &direct_data3_;};
        vector_t *global_data() {return &global_data_;};
        vector_t *global_data2() {return &global_data2_;};
        vector<int> *mapping() {return &mapping_;};
        bool IsCouplingOngoing() {return precice_->isCouplingOngoing();};

        ~Coupling() {Finalize();}

};


#endif