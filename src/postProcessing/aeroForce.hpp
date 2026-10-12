#if !defined(nekrs_aeroforce_hpp_)
#define nekrs_aeroforce_hpp_

#include <array>
#include "nrssys.hpp"

// Ported from upstream nekRS (src/app/nrs/postProcessing/aeroForce.hpp, v24+), without the
// normal/tangential split of the viscous force (upstream builds it from viscousShearStress /
// viscousNormalStress, which v23 does not have).
class AeroForce
{
public:
  AeroForce() {};

  std::array<dfloat, 3> viscous() const { return forceV; };
  void setViscousForce(std::array<dfloat, 3> f) { forceV = f; }

  std::array<dfloat, 3> pressure() const { return forceP; };
  void setPressureForce(std::array<dfloat, 3> f) { forceP = f; }

  std::array<dfloat, 3> total() const
  {
    return {forceV[0] + forceP[0], forceV[1] + forceP[1], forceV[2] + forceP[2]};
  };

private:
  std::array<dfloat, 3> forceV = {0, 0, 0};
  std::array<dfloat, 3> forceP = {0, 0, 0};
};

#endif
