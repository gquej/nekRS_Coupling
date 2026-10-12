#include "nrs.hpp"
#include "platform.hpp"
#include "postProcessing.hpp"

namespace {
  std::vector<dfloat> sum;
  occa::memory o_sum;
}

// Ported from upstream nekRS (nrs_t::aeroForces, v24+): force exerted by the fluid on the
// faces whose boundary ID is in o_bID (nbID entries), split into its viscous and pressure parts.
// o_Sij must hold the strain rate (postProcessing::strainRate layout, 6 fields).
AeroForce postProcessing::aeroForces(nrs_t *nrs, int nbID, const occa::memory& o_bID, const occa::memory& o_Sij)
{
  mesh_t *mesh = nrs->meshV;

  if (o_sum.size() == 0) {
    sum.resize(6 * mesh->Nelements, 0);
    o_sum = platform->device.malloc(sum.size() * sizeof(dfloat), sum.data());
  }

  auto kernel = platform->kernels.get("aeroForces");

  kernel(mesh->Nelements,
         nrs->fieldOffset,
         nbID,
         o_bID,
         mesh->o_sgeo,
         mesh->o_vmapM,
         mesh->o_EToB,
         nrs->o_mue,
         nrs->o_P,
         o_Sij,
         o_sum);

  o_sum.copyTo(sum.data(), sum.size() * sizeof(dfloat));

  dfloat f[6] = {0, 0, 0, 0, 0, 0};
  for (int k = 0; k < 6; k++) {
    for (dlong e = 0; e < mesh->Nelements; e++) {
      f[k] += sum[e + k * mesh->Nelements];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, f, 6, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);

  AeroForce af;
  af.setViscousForce({f[0], f[1], f[2]});
  af.setPressureForce({f[3], f[4], f[5]});
  return af;
}
