#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "postProcessing.hpp"

namespace {
  dfloat *drag;
  dfloat *dragx;
  dfloat *dragy;
  dfloat *dragz;
  occa::memory o_drag;
  occa::memory o_dragx;
  occa::memory o_dragy;
  occa::memory o_dragz;
}

dfloat postProcessing::viscousDrag(nrs_t *nrs, int nbID, const occa::memory& o_bID, occa::memory& o_Sij)
{
  mesh_t *mesh = nrs->meshV;

  if(o_drag.size() == 0) { 
    drag = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    o_drag = platform->device.malloc(mesh->Nelements * sizeof(dfloat), drag);
  } 

  auto dragKernel = platform->kernels.get("drag");
  
  dragKernel(mesh->Nelements,
	     nrs->fieldOffset,
	     nbID,
	     o_bID,
	     mesh->o_sgeo,
	     mesh->o_vmapM,
	     mesh->o_EToB,
	     nrs->o_mue,
	     o_Sij,
	     o_drag);

  o_drag.copyTo(drag);

  dfloat sum = 0;
  for (dlong i = 0; i < mesh->Nelements; i++) {
    sum += drag[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
    
  return sum;
}

void postProcessing::totalDragVector(nrs_t *nrs, int nbID, const occa::memory& o_bID, occa::memory& o_Sij, float dragV[3])
{
  mesh_t *mesh = nrs->meshV;

  if(o_dragx.size() == 0) { 
    dragx = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    dragy = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    dragz = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    o_dragx = platform->device.malloc(mesh->Nelements * sizeof(dfloat), dragx);
    o_dragy = platform->device.malloc(mesh->Nelements * sizeof(dfloat), dragy);
    o_dragz = platform->device.malloc(mesh->Nelements * sizeof(dfloat), dragz);
  } 
  
  //auto o_Sij = platform->device.malloc(2 * nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));
  //postProcessing::strainRate(nrs, false, o_Sij);  // fills stress tensor Sij

  //std::vector<int> bidWall = {nbID};
  //occa::memory o_bID = platform->device.malloc(bidWall.size(), bidWall.data());

  auto dragVectorKernel = platform->kernels.get("dragVector");
  
  dragVectorKernel(mesh->Nelements,
       nrs->fieldOffset,
       nbID,
       o_bID,
       mesh->o_sgeo,
       mesh->o_vmapM,
       mesh->o_EToB,
       nrs->o_mue,
       o_Sij,
       nrs->o_P,
       o_dragx,
       o_dragy,
       o_dragz);

  o_dragx.copyTo(dragx);
  o_dragy.copyTo(dragy);
  o_dragz.copyTo(dragz);

  dfloat sumx = 0.0;
  dfloat sumy = 0.0;
  dfloat sumz = 0.0;
  for (dlong i = 0; i < mesh->Nelements; i++) {
    sumx += dragx[i];
    sumy += dragy[i];
    sumz += dragz[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, &sumx, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  MPI_Allreduce(MPI_IN_PLACE, &sumy, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  MPI_Allreduce(MPI_IN_PLACE, &sumz, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);

  dragV[0] = sumx*2.0;
  dragV[1] = sumy*2.0;
  dragV[2] = sumz*2.0;
  //o_Sij.free();
}


