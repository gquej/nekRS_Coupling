#ifndef COUPLING_HPP
#define COUPLING_HPP


//#include "nrs.hpp"

#include <iostream>
#include <algorithm> // for std::copy
#include <occa.hpp>
class Coupling {
    protected:
        double * vertices_coord_;
        int * vertices_mapping_;
        double * data_;
        double foo_;

        occa::memory o_data;
        occa::memory o_mapping;
    public: 
        Coupling() = default;
        explicit Coupling(double foo): foo_(foo) {};
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
        

        ~Coupling() {
            delete[] vertices_coord_;
            delete[] vertices_mapping_;
            delete[] data_;
        }



};


#endif