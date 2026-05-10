#include "nrs.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "postProcessing.hpp"

namespace {
  dfloat *drag;
  dfloat *dragx;
  dfloat *dragy;
  dfloat *dragz;
  dfloat *trqx;
  dfloat *trqy;
  dfloat *trqz;
  occa::memory o_drag;
  occa::memory o_dragx;
  occa::memory o_dragy;
  occa::memory o_dragz;
  occa::memory o_trqx;
  occa::memory o_trqy;
  occa::memory o_trqz;
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

void postProcessing::totalDragVector(nrs_t *nrs, int nbID, const occa::memory& o_bID, occa::memory& o_Sij, float dragV[6], occa::memory& o_pos)
{
  mesh_t *mesh = nrs->meshV;

  if(o_dragx.size() == 0) { 
    dragx = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    dragy = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    dragz = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    trqx  = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    trqy  = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    trqz  = (dfloat *) calloc(mesh->Nelements, sizeof(dfloat));
    o_dragx = platform->device.malloc(mesh->Nelements * sizeof(dfloat), dragx);
    o_dragy = platform->device.malloc(mesh->Nelements * sizeof(dfloat), dragy);
    o_dragz = platform->device.malloc(mesh->Nelements * sizeof(dfloat), dragz);
    o_trqx  = platform->device.malloc(mesh->Nelements * sizeof(dfloat), trqx);
    o_trqy  = platform->device.malloc(mesh->Nelements * sizeof(dfloat), trqy);
    o_trqz  = platform->device.malloc(mesh->Nelements * sizeof(dfloat), trqz);
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
       o_dragz,
       o_trqx,
       o_trqy,
       o_trqz,
       mesh->o_x,
       mesh->o_y,
       mesh->o_z,
       o_pos);

  o_dragx.copyTo(dragx);
  o_dragy.copyTo(dragy);
  o_dragz.copyTo(dragz);
  o_trqx.copyTo(trqx);
  o_trqy.copyTo(trqy);
  o_trqz.copyTo(trqz);

  dfloat sumx = 0.0;
  dfloat sumy = 0.0;
  dfloat sumz = 0.0;
  dfloat sumTrqx = 0.0;
  dfloat sumTrqy = 0.0;
  dfloat sumTrqz = 0.0;

  for (dlong i = 0; i < mesh->Nelements; i++) {
    sumx += dragx[i];
    sumy += dragy[i];
    sumz += dragz[i];
    sumTrqx += trqx[i];
    sumTrqy += trqy[i];
    sumTrqz += trqz[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, &sumx, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  MPI_Allreduce(MPI_IN_PLACE, &sumy, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  MPI_Allreduce(MPI_IN_PLACE, &sumz, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  MPI_Allreduce(MPI_IN_PLACE, &sumTrqx, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  MPI_Allreduce(MPI_IN_PLACE, &sumTrqy, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  MPI_Allreduce(MPI_IN_PLACE, &sumTrqz, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);

  dragV[0] = sumx*1.0; // Assume dimensionless force (F/rhoU²S) => multiply by 2 to get drag coefficient
  dragV[1] = sumy*1.0;
  dragV[2] = sumz*1.0;
  dragV[3] = sumTrqx*1.0;
  dragV[4] = sumTrqy*1.0;
  dragV[5] = sumTrqz*1.0;
  //o_Sij.free();
}

void postProcessing::integrateDynamics(nrs_t *nrs, float dragV[6]) {
  /*
  
  pos : x, y, z coordinates of the center of mass of the body at tn
  orientation : phi, theta, psi angles at tn
  vel : velocity vector at tn
  omega : angular velocity vector at tn
  dragV : drag force and torque vector at tn+1 (dragV[0], dragV[1], dragV[2] are force components, dragV[3], dragV[4], dragV[5] are torque components)
  => Everything is in inertial frame !
  
  */
  
  mesh_t *mesh = nrs->meshV;
  dfloat * pos = nrs->position_new;
  dfloat * orientation = nrs->orientation_new;
  dfloat * vel = nrs->velocity;
  dfloat * omega = nrs->omega;
  float * oldforce = nrs->oldforce;
  float m  = nrs->MASS;
  float I1 = nrs->INERTIA[0];   // TBD in setup !
  float I2 = nrs->INERTIA[1];
  float I3 = nrs->INERTIA[2];
  dfloat dt = nrs->dt[0];
  for (int i = 0; i < 3; i++) {
    //pos[i] += vel[i] * dt;         // Explicit Euler for position update (use current velocity)
    dfloat oldvel = vel[i];
    vel[i] += (3.0*dragV[i] - oldforce[i]) / (2.0*m) * dt;   // Implicit Euler (F is at n+1 as flow is at n+1 in post-processing) or Explicit if oldforce
    pos[i] += (oldvel + vel[i]) / 2.0 * dt;         // Implicit Euler for position update (use updated velocity)
  } 
  std::vector<std::vector<dfloat>> Q(3, std::vector<dfloat>(3));
  dfloat phi = orientation[0];
  dfloat theta = orientation[1];
  dfloat psi = orientation[2];

  Q[0][0] = cos(theta)*cos(psi);
  Q[0][1] = cos(theta)*sin(psi);
  Q[0][2] = -sin(theta);
  Q[1][0] = sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi);
  Q[1][1] = sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi);
  Q[1][2] = sin(phi)*cos(theta);
  Q[2][0] = cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi);
  Q[2][1] = cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi);
  Q[2][2] = cos(phi)*cos(theta);

  /*
  double cp = cos(psi / 2.0);
  double sp = sin(psi / 2.0);
  double ct = cos(theta / 2.0);
  double st = sin(theta / 2.0);
  double cf = cos(phi / 2.0);
  double sf = sin(phi / 2.0);
  
  q[0] = cp * ct * cf + sp * st * sf;  // w
  q[1] = cp * ct * sf - sp * st * cf;  // x
  q[2] = cp * st * cf + sp * ct * sf;  // y
  q[3] = sp * ct * cf - cp * st * sf;  // z
  
  // Normalize
  double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
  for (int i = 0; i < 4; i++) q[i] /= norm;

  // Change w of basis with q w q^-1 and integrate with torque
  // Bring it back q^1 w q  for wx_n+1
  // Integrate quaternion in time (maybe only with wx_n)
  // Find psi, theta, phi from quaternion
  // psi (rotation autour Z)
  psi = atan2(2.0*(q0*q3 + q1*q2), 1.0 - 2.0*(q2*q2 + q3*q3));
  
  // theta (rotation autour Y)
  double sinTheta = 2.0*(q0*q2 - q3*q1);
  sinTheta = (sinTheta > 1.0) ? 1.0 : ((sinTheta < -1.0) ? -1.0 : sinTheta);
  theta = asin(sinTheta);
  
  // phi (rotation autour X)
  phi = atan2(2.0*(q0*q1 + q2*q3), 1.0 - 2.0*(q1*q1 + q2*q2));

  */

  float trqBody[3] = {0.0, 0.0, 0.0};
  float omegaBody[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      trqBody[i] += Q[i][j] * (3.0*dragV[j + 3] - oldforce[j + 3])/2.0; // dragV[3], dragV[4], dragV[5] are torque components
      omegaBody[i] += Q[i][j] * omega[j]; // Transform angular velocity to body frame
    }
  }
  float domega_dt[3] = {0.0, 0.0, 0.0};
  domega_dt[0] = (trqBody[0] - (I3 - I2)*omegaBody[1]*omegaBody[2]) / I1;
  domega_dt[1] = (trqBody[1] - (I1 - I3)*omegaBody[2]*omegaBody[0]) / I2;
  domega_dt[2] = (trqBody[2] - (I2 - I1)*omegaBody[0]*omegaBody[1]) / I3;
  for (int i = 0; i < 3; i++) {
    omegaBody[i] += domega_dt[i] * dt;
  }
  
  float dphi_dt = omegaBody[0] + sin(phi)*tan(theta)*omegaBody[1] + cos(phi)*tan(theta)*omegaBody[2];
  float dtheta_dt = cos(phi)*omegaBody[1] - sin(phi)*omegaBody[2];
  float dpsi_dt = sin(phi)/cos(theta)*omegaBody[1] + cos(phi)/cos(theta)*omegaBody[2];
  orientation[0] += dphi_dt * dt;
  orientation[1] += dtheta_dt * dt;
  orientation[2] += dpsi_dt * dt;

  for (int i = 0; i < 3; i++){
    omega[i] = 0.0;
    for (int j = 0; j < 3; j++) {
      omega[i] += Q[j][i] * omegaBody[j]; // Transform back to inertial frame
    }
  }
  for (int i = 0; i < 6; i++) {
    nrs->oldforce[i] = dragV[i];   // For Euler explicit or Crank-Nicolson
  }

  nrs->o_position.copyFrom(nrs->position_new);
  nrs->o_orientation.copyFrom(nrs->orientation_new);
  nrs->o_velocity.copyFrom(nrs->velocity);
  nrs->o_omega.copyFrom(nrs->omega);
}