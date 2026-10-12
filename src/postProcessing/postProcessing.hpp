#if !defined(nekrs_post_hpp_)
#define nekrs_post_hpp_

#include "nrs.hpp"
#include "aeroForce.hpp"

namespace postProcessing
{

void planarAvg(nrs_t *nrs, const std::string& dir, int NELGX, int NELGY, int NELGZ, int nflds, occa::memory o_avg);
// Force of the fluid on the faces with a boundary ID in o_bID (nbID entries, device ints);
// o_Sij from strainRate() below.
AeroForce aeroForces(nrs_t *nrs, int nbID, const occa::memory& o_bID, const occa::memory& o_Sij);

//       ( SO0          )         (     SO8  SO7)
// Sij = ( SO3  SO1     )  Oij =  (          SO6)
//       ( SO5  SO4  SO2)         (             )
void strainRotationRate(nrs_t *nrs, bool smooth, bool rotationRate, occa::memory& o_Sij);
void strainRate(nrs_t *nrs, bool smooth, occa::memory& o_Sij);

void Qcriterion(nrs_t *nrs, occa::memory& o_Q);
}

#endif
