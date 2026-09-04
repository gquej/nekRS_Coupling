#include <stdlib.h>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <functional>
#include "nrs.hpp"
#include "pointInterpolation.hpp"
#include <cmath>
#include "setup.hpp"
#include "nekInterfaceAdapter.hpp"
#include "printHeader.hpp"
#include "udf.hpp"
#include "bcMap.hpp"
#include "parReader.hpp"
#include "re2Reader.hpp"
#include "configReader.hpp"
#include "re2Reader.hpp"
#include "timeStepper.hpp"
#include "platform.hpp"
#include "linAlg.hpp"
#include "cfl.hpp"
#include "AMGX.hpp"
#include "hypreWrapper.hpp"
#include "hypreWrapperDevice.hpp"

namespace fs = std::filesystem;

// extern variable from nrssys.hpp
platform_t *platform;

static nrs_t *nrs;
static setupAide options;

static int rank, size;
static MPI_Comm commg, comm;

static dfloat lastOutputTime = 0;
static int firstOutfld = 1;
static int enforceLastStep = 0;
static int enforceOutputStep = 0;
static bool initialized = false;
static int nekDumpCount = 0; // classic dumps written by this process
static int nekChkCount = 0;  // restart checkpoints written by this process
static bool firstFldDump = true; // first classic dump of this process
static bool firstChkDump = true; // first restart checkpoint of this process
static std::string couplingDir; // Coupling_dir, taken from the preciceConfig path

namespace nekrs {

void reset()
{
  lastOutputTime = 0;
  firstOutfld = 1;
  enforceLastStep = 0;
  enforceOutputStep = 0;
  nekDumpCount = 0;
  nekChkCount = 0;
  firstFldDump = true;
  firstChkDump = true;
}

double startTime(void)
{
  double val = 0;
  platform->options.getArgs("START TIME", val);
  return val;
}



void setup(MPI_Comm commg_in,
           MPI_Comm comm_in,
           int buildOnly,
           int commSizeTarget,
           int ciMode,
           std::string _setupFile,
           std::string _backend,
           std::string _deviceID,
           int nSessions,
           int sessionID,
           int debug)
{
  nrsCheck(initialized, comm_in, EXIT_FAILURE, "%s\n", "Calling setup twice is erroneous!");

  commg = commg_in;
  comm = comm_in;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == 0) {
    printHeader();
    std::cout << "MPI tasks: " << size << std::endl << std::endl;
  }

  configRead(comm);

  if(nSessions > 1) {
    options.setArgs("NEKNEK NUMBER OF SESSIONS", std::to_string(nSessions));
    options.setArgs("NEKNEK SESSION ID", std::to_string(sessionID));
  }

  options.setArgs("BUILD ONLY", "FALSE");
  if (buildOnly) {
    options.setArgs("BUILD ONLY", "TRUE");
    options.setArgs("NP TARGET", std::to_string(commSizeTarget));
    if (rank == 0) {
      std::cout << "jit-compiling for >=" << commSizeTarget << " MPI tasks ...\n" << std::endl;
    }
    fflush(stdout);
  }

  auto par = new inipp::Ini();
  if (rank == 0)
    std::cout << "reading par file ...\n";
  parRead(par, _setupFile + ".par", comm, options);

  // precedence: cmd arg, par, env-var
  if (options.getArgs("THREAD MODEL").length() == 0)
    options.setArgs("THREAD MODEL", getenv("NEKRS_OCCA_MODE_DEFAULT"));
  if (!_backend.empty())
    options.setArgs("THREAD MODEL", _backend);
  if (!_deviceID.empty())
    options.setArgs("DEVICE NUMBER", _deviceID);

  // setup platform (requires THREAD MODEL)
  platform_t *_platform = platform_t::getInstance(options, commg, comm);
  platform = _platform;
  platform->par = par;

  if (debug)
    platform->options.setArgs("VERBOSE", "TRUE");

  int buildRank = rank;
  if (platform->cacheLocal)
    MPI_Comm_rank(platform->comm.mpiCommLocal, &buildRank);

  if (rank == 0) {
    std::cout << "using NEKRS_HOME: " << getenv("NEKRS_HOME") << std::endl;
    std::cout << "using NEKRS_CACHE_DIR: " << getenv("NEKRS_CACHE_DIR") << std::endl;
    std::cout << "using OCCA_CACHE_DIR: " << occa::env::OCCA_CACHE_DIR << std::endl << std::endl;
  }

  options.setArgs("CI-MODE", std::to_string(ciMode));
  if (rank == 0 && ciMode)
    std::cout << "enabling continous integration mode " << ciMode << "\n";

  {
    int nelgt, nelgv;
    re2::nelg(options.getArgs("MESH FILE"), nelgt, nelgv, comm);
    nrsCheck(size > nelgv, platform->comm.mpiComm, EXIT_FAILURE, "%s\n", "MPI tasks > number of elements!");
  }

  bcMap::setup();

  nek::bootstrap();

  // jit compile udf
  std::string udfFile;
  options.getArgs("UDF FILE", udfFile);
  if (!udfFile.empty()) {
    udfBuild(udfFile, options);
    udfLoad();
  }

  // here we might access some nek variables
  if (udf.setup0)
    udf.setup0(comm, options);

  if (rank == 0) {
    if (!buildOnly && options.compareArgs("STDOUT PAR", "TRUE"))
      parEcho();
    if (!buildOnly && options.compareArgs("STDOUT UDF", "TRUE"))
      udfEcho();
  }

  compileKernels();

  oogs::overlap(options.compareArgs("ENABLE GS COMM OVERLAP", "FALSE") ? 0 : 1);

  if (buildOnly) {
    MPI_Barrier(platform->comm.mpiComm);
    if (buildRank == 0) {
      std::string cache_dir;
      cache_dir.assign(getenv("NEKRS_CACHE_DIR"));
      std::string file = cache_dir + "/build-only.timestamp";
      remove(file.c_str());
      std::ofstream ofs;
      ofs.open(file, std::ofstream::out);
      ofs.close();
      if (rank == 0)
        std::cout << "\nBuild successful." << std::endl;
    }
    return;
  }

  platform->linAlg = linAlg_t::getInstance();

  nrs = new nrs_t();

  {
    int result = 0;
    MPI_Comm_compare(commg, comm, &result);

    nrs->multiSession = (result == MPI_UNEQUAL);
  }

  nrsSetup(comm, options, nrs);
  if (neknekCoupled()) {
    new neknek_t(nrs, nSessions, sessionID);
  }

  const double setupTime = platform->timer.query("setup", "DEVICE:MAX");
  if (rank == 0) {
    std::cout << "\nsettings:\n" << std::endl << options << std::endl;
    std::cout << "occa memory usage: " << platform->device.occaDevice().memoryAllocated() / 1e9 << " GB"
              << std::endl;
  }
  fflush(stdout);

  platform->flopCounter->clear();

#if 1
  if (platform->cacheBcast) {
    MPI_Barrier(platform->comm.mpiComm);

    int rankLocal;
    MPI_Comm_rank(platform->comm.mpiCommLocal, &rankLocal);
    if (rankLocal == 0) {
      for (auto &entry : std::filesystem::directory_iterator(platform->tmpDir))
        fs::remove_all(entry.path());
    }
  }
#endif

  initialized = true;
}
void couplingSetup(std::string_view config_file,std::string_view solver_name,
  std::string_view mesh_name, std::string_view interior_mesh_name, std::string_view direct_mesh_name,
  std::string_view data_name, std::string_view data2_name,
  std::string_view direct_data_name, std::string_view direct_data_name_cum,
  double tol_bb, bool *periodic_dir, double * periodic_bounds,
  int M_VPM, bool staggered, bool pre_simulation, int pre_sim_expected_last_step) {
  mesh_t *mesh = nrs->meshV;
  int p_Nfaces = mesh->Nfaces;
  int p_Nfp = mesh->Nfp;
  int p_Np = mesh->Np;
  int p_Nq = mesh->Nq;
  int N_elements = mesh->Nelements;
  nrs->coupling_pre_simulation = pre_simulation;
  nrs->coupling_pre_sim_expected_last_step = pre_sim_expected_last_step;
  double * nek_x = mesh->x;
  double * nek_y = mesh->y;
  double * nek_z = mesh->z;
  dfloat bbx1 = 1e30, bbx2 = -1e30;
  dfloat bby1 = 1e30, bby2 = -1e30;
  dfloat bbz1 = 1e30, bbz2 = -1e30;
  for (dlong e = 0; e < N_elements; e ++) {
    for (int k = 0; k < p_Nq; ++k) {
      for (int j = 0; j < p_Nq; ++j) {
        for (int i = 0; i < p_Nq; ++i) {
          const dlong id = e * p_Np + k * p_Nq * p_Nq + j * p_Nq + i;
          if (nek_x[id] < bbx1) bbx1 = nek_x[id];
          if (nek_x[id] > bbx2) bbx2 = nek_x[id];
          if (nek_y[id] < bby1) bby1 = nek_y[id];
          if (nek_y[id] > bby2) bby2 = nek_y[id];
          if (nek_z[id] < bbz1) bbz1 = nek_z[id];
          if (nek_z[id] > bbz2) bbz2 = nek_z[id];
        }
      }
    }
  }

  nrs->coupling_bbox = new double[6];
  nrs->coupling_bbox[0] = bbx1 - tol_bb;
  nrs->coupling_bbox[1] = bbx2 + tol_bb;
  nrs->coupling_bbox[2] = bby1 - tol_bb;
  nrs->coupling_bbox[3] = bby2 + tol_bb;
  nrs->coupling_bbox[4] = bbz1 - tol_bb;
  nrs->coupling_bbox[5] = bbz2 + tol_bb;

  nrs->coupling = new Coupling(solver_name, config_file);
  Coupling * coupling = nrs->coupling;

  int boundary_points_counter = 0;
  int interior_boundary_points_counter = 0;

  nrs->coupling_vmap = (dlong *) calloc(mesh->Nelements * mesh->Np, sizeof(dlong)); //mapping from volumic indices to precice buffer (i.e., a lot of these are empty since a lot of nodes are not at a boundary)
  for (int i = 0; i < mesh->Nelements * mesh->Np; i++)
  {
    nrs->coupling_vmap[i] = -1; //default: no mapping
  }
  
  //first loop to count the total number of vertices on the outer nek boundary (this includes doubles)
  for (dlong e = 0; e < N_elements; e++) {
    for (int f = 0; f < p_Nfaces; f++) {
      const dlong bcType = nrs->EToB[f + p_Nfaces * e];
      if (bcType == 3) {
        boundary_points_counter += p_Nfp * 2; //factor 2 is to count also the interior layer of points when imposing omega on the second spectral point
      } else if (bcType == 1) {
        interior_boundary_points_counter += p_Nfp;
      }
    }
  }
  double vertices_temp [3* boundary_points_counter]; //temp array to store all the outer vertices
  coupling->Resize_mapping(boundary_points_counter); //total number of nek outer vertices (as seen by nek)
  std::vector<int> * mapping = coupling->mapping(); //mapping from the nek mesh to the precice buffe
  int counter_to_idM[boundary_points_counter];

  double interior_vertices_temp [3* interior_boundary_points_counter]; //temp array to store all the inner vertices
  int interior_counter_to_idM[interior_boundary_points_counter];
  //filling vertices_temp with all the outer vertices (including the doubles!!)

  int total_count = 0;
  int interior_total_count = 0;

  for (dlong e = 0; e < N_elements; e++) {
    for (int f = 0; f < p_Nfaces; f++) {
      const dlong bcType = nrs->EToB[f + p_Nfaces * e];
      if (bcType == 3) {
        for (int m = 0; m < p_Nfp; ++m) {
          const int n = m + f * p_Nfp;
          const int sk = e * p_Nfp * p_Nfaces + n;
          const dlong idM = ((mesh->vmapM))[sk];

          if (periodic_dir[0] && abs(nek_x[idM] - periodic_bounds[1]) < 1e-8) {           //periodicity in x hardcoded here
            vertices_temp[3 * total_count + 0] = periodic_bounds[0];
          } else {
            vertices_temp[3 * total_count + 0] = nek_x[idM];
          }

          if (periodic_dir[1] && abs(nek_y[idM] - periodic_bounds[3]) < 1e-8) {           //periodicity in y hardcoded here
            vertices_temp[3 * total_count + 1] = periodic_bounds[2];
          } else {
            vertices_temp[3 * total_count + 1] = nek_y[idM];
          }

          if (periodic_dir[2] && abs(nek_z[idM] - periodic_bounds[5]) < 1e-8) {           //periodicity in z hardcoded here
            vertices_temp[3 * total_count + 2] = periodic_bounds[4];
          } else {
            vertices_temp[3 * total_count + 2] = nek_z[idM];
          }

          counter_to_idM[total_count] = idM;
          total_count ++;


          //sending the second layer of points also 
          const dlong idM_el = idM - e * p_Np ;               // equivalent to k * p_Nq * p_Nq + j * p_Nq + i
          const dlong remainder = idM_el % (p_Nq * p_Nq);            
          const int idz = idM_el/(p_Nq * p_Nq);             //volumic z (k) index within this element 
          const int idy = remainder / p_Nq;                 //volumic y (j) index within this element 
          const int idx = remainder % p_Nq;
          dlong idM_local;
          if (f == 4) idM_local = e * p_Np + idz * p_Nq * p_Nq + idy * p_Nq + idx + 1; 
          if (f == 2) idM_local = e * p_Np + idz * p_Nq * p_Nq + idy * p_Nq + idx - 1; 
          if (f == 1) idM_local = e * p_Np + idz * p_Nq * p_Nq + (idy + 1) * p_Nq + idx; 
          if (f == 3) idM_local = e * p_Np + idz * p_Nq * p_Nq + (idy - 1) * p_Nq + idx; 
          if (f == 0) idM_local = e * p_Np + (idz + 1) * p_Nq * p_Nq + idy * p_Nq + idx; 
          if (f == 5) idM_local = e * p_Np + (idz - 1) * p_Nq * p_Nq + idy * p_Nq + idx; 

          if (periodic_dir[0] && abs(nek_x[idM_local] - periodic_bounds[1]) < 1e-8) {           //periodicity in x hardcoded here
            vertices_temp[3 * total_count + 0] = periodic_bounds[0];
          } else {
            vertices_temp[3 * total_count + 0] = nek_x[idM_local];
          }

          if (periodic_dir[1] && abs(nek_y[idM_local] - periodic_bounds[3]) < 1e-8) {           //periodicity in y hardcoded here
            vertices_temp[3 * total_count + 1] = periodic_bounds[2];
          } else {
            vertices_temp[3 * total_count + 1] = nek_y[idM_local];
          }

          if (periodic_dir[2] && abs(nek_z[idM_local] - periodic_bounds[5]) < 1e-8) {           //periodicity in z hardcoded here
            vertices_temp[3 * total_count + 2] = periodic_bounds[4];
          } else {
            vertices_temp[3 * total_count + 2] = nek_z[idM_local];
          }
          counter_to_idM[total_count] = idM_local;
          total_count ++;
        }
      } else if (bcType == 1) {
        for (int m = 0; m < p_Nfp; ++m) {
          const int n = m + f * p_Nfp;
          const int sk = e * p_Nfp * p_Nfaces + n;
          const dlong idM = ((mesh->vmapM))[sk];

          if (periodic_dir[0] && abs(nek_x[idM] - periodic_bounds[1]) < 1e-8) {           //periodicity in x hardcoded here
            interior_vertices_temp[3 * interior_total_count + 0] = periodic_bounds[0];
          } else {
            interior_vertices_temp[3 * interior_total_count + 0] = nek_x[idM];
          }

          if (periodic_dir[1] && abs(nek_y[idM] - periodic_bounds[3]) < 1e-8) {           //periodicity in y hardcoded here
            interior_vertices_temp[3 * interior_total_count + 1] = periodic_bounds[2];
          } else {
            interior_vertices_temp[3 * interior_total_count + 1] = nek_y[idM];
          }

          if (periodic_dir[2] && abs(nek_z[idM] - periodic_bounds[5]) < 1e-8) {           //periodicity in z hardcoded here
            interior_vertices_temp[3 * interior_total_count + 2] = periodic_bounds[4];
          } else {
            interior_vertices_temp[3 * interior_total_count + 2] = nek_z[idM];
          }
          interior_counter_to_idM[interior_total_count] = idM;
          interior_total_count ++;
        }
      }
    }
  }

  //filtering all the outer nek vertices that are duplicated across this rank (and only this rank)
  int unique_count = 0;
  double diff1, diff2, diff3;
  double tol = 1.e-10;
  for (int i = 0; i < boundary_points_counter; i++)
  {
    const dlong idM = counter_to_idM[i];
    int is_unique = 1;
    for (int j = 0; j < unique_count; j++)
    {
      diff1 = fabs(vertices_temp[3 * i + 0] - vertices_temp[3 * j + 0]);
      diff2 = fabs(vertices_temp[3 * i + 1] - vertices_temp[3 * j + 1]);
      diff3 = fabs(vertices_temp[3 * i + 2] - vertices_temp[3 * j + 2]);
      if ((diff1 < tol) && (diff2 < tol) && (diff3 < tol) && (is_unique == 1)) {
        is_unique = 0;
        (*mapping)[i] = j;
        nrs->coupling_vmap[idM] = j;
      }
    }
    if (is_unique == 1) {
      vertices_temp[3 * unique_count + 0] = vertices_temp[3 * i + 0];
      vertices_temp[3 * unique_count + 1] = vertices_temp[3 * i + 1];
      vertices_temp[3 * unique_count + 2] = vertices_temp[3 * i + 2];
      (*mapping)[i] = unique_count;
      nrs->coupling_vmap[idM] = unique_count;
      unique_count ++;
    }
  }

  coupling->Set_vertices(vertices_temp, unique_count);
  coupling->Set_interior_vertices(interior_vertices_temp, interior_total_count);

  nrs->o_coupling_data1 = platform->device.malloc(sizeof(double) * unique_count * 3, coupling ->Get_data1());
  nrs->o_coupling_data2 = platform->device.malloc(sizeof(double) * unique_count * 3, coupling ->Get_data2());
  nrs->o_coupling_mapping = platform->device.malloc(sizeof(int) * boundary_points_counter, coupling ->Get_mapping());

  nrs->o_coupling_data1.copyFrom(coupling->Get_data1());
  nrs->o_coupling_data2.copyFrom(coupling->Get_data2());
  nrs->o_coupling_mapping.copyFrom(coupling->Get_mapping());

  nrs->o_coupling_vmap = platform->device.malloc(mesh->Nelements * mesh->Np * sizeof(dlong), nrs->coupling_vmap);
  nrs->o_coupling_vmap.copyFrom(nrs->coupling_vmap);

  nrs->o_coupling_bbox = platform->device.malloc(6 * sizeof(dlong), nrs->coupling_bbox);
  nrs->o_coupling_bbox.copyFrom(nrs->coupling_bbox);

  const int NfpTotal = mesh->Nelements * mesh->Nfaces * mesh->Nfp;
  const int Nblock = (NfpTotal + BLOCKSIZE - 1) / BLOCKSIZE;
  nrs->coupling_flux = (dfloat *)calloc(NfpTotal, sizeof(dfloat));
  nrs->coupling_area = (dfloat *)calloc(NfpTotal, sizeof(dfloat));
  nrs->coupling_tmp1 = (dfloat *)calloc(Nblock, sizeof(dfloat));
  nrs->coupling_tmp2 = (dfloat *)calloc(Nblock, sizeof(dfloat));
  nrs->o_coupling_flux = platform->device.malloc(NfpTotal * sizeof(dfloat), nrs->coupling_flux);
  nrs->o_coupling_area = platform->device.malloc(NfpTotal * sizeof(dfloat), nrs->coupling_area);
  nrs->o_coupling_tmp1 = platform->device.malloc(Nblock * sizeof(dfloat), nrs->coupling_tmp1);
  nrs->o_coupling_tmp2 = platform->device.malloc(Nblock * sizeof(dfloat), nrs->coupling_tmp2);

  coupling->Setup(mesh_name, interior_mesh_name, direct_mesh_name, data_name, nrs->coupling_bbox, data2_name, direct_data_name, direct_data_name_cum, staggered, M_VPM);
  
  //setting up the interpolator

  const int np = coupling->direct_mesh_size();
  const auto offset = np;
  nrs->interpolator = new pointInterpolation_t(nrs);
  int n_VPM_mesh;
  if (staggered) n_VPM_mesh = 3;
  else n_VPM_mesh = 1;
  nrs->o_fields1D = platform->device.malloc(n_VPM_mesh * 3 * offset * sizeof(dfloat));
  std::vector<dfloat> xp, yp, zp;
  
  const std::vector<dfloat> *vertices = coupling->direct_vertices();
  const double h_VPM =  1. /((double) M_VPM);
  if (staggered) {
    // for u
    for (int i = 0; i < np; i++) {
      xp.push_back((*vertices)[3 * i + 0]);
      yp.push_back((*vertices)[3 * i + 1] + 0.5 * h_VPM);
      zp.push_back((*vertices)[3 * i + 2] + 0.5 * h_VPM);
    }
    // for v
    for (int i = 0; i < np; i++) {
      xp.push_back((*vertices)[3 * i + 0] + 0.5 * h_VPM);
      yp.push_back((*vertices)[3 * i + 1]);
      zp.push_back((*vertices)[3 * i + 2] + 0.5 * h_VPM);
    }
    // for w
    for (int i = 0; i < np; i++) {
      xp.push_back((*vertices)[3 * i + 0] + 0.5 * h_VPM);
      yp.push_back((*vertices)[3 * i + 1] + 0.5 * h_VPM);
      zp.push_back((*vertices)[3 * i + 2]);
    }
    
  } else {
    for (int i = 0; i < np; i++) {
      xp.push_back((*vertices)[3 * i + 0]);
      yp.push_back((*vertices)[3 * i + 1]);
      zp.push_back((*vertices)[3 * i + 2]);
    }
  }

  nrs->interpolator->setPoints(n_VPM_mesh * np, xp.data(), yp.data(), zp.data());
  nrs->interpolator->find();
  // the init loop just below writes n_VPM_mesh * fieldOffset entries; allocating only
  // fieldOffset overran the heap by 3x on the staggered path
  nrs->coupling_cum = (dfloat *)calloc(n_VPM_mesh * nrs->fieldOffset, sizeof(dfloat));
  nrs->o_coupling_cum = platform->device.malloc(nrs->fieldOffset * sizeof(dfloat), nrs->coupling_cum);
  for (int i = 0; i < n_VPM_mesh * nrs->fieldOffset; i++) {
    nrs->coupling_cum[i] = 1.0;
  }
  nrs->o_coupling_cum.copyFrom(nrs->coupling_cum);
  nrs->o_fields1D_cum = platform->device.malloc(n_VPM_mesh * offset * sizeof(dfloat));  
}

void couplingFinalize() {
  delete nrs->coupling;
  free(nrs->coupling_vmap);
  nrs->o_coupling_data1.free();
  nrs->o_coupling_data2.free();
  nrs->o_coupling_mapping.free();
  nrs->o_coupling_vmap.free();
  nrs->o_coupling_bbox.free();
  delete nrs->interpolator;
  nrs->o_fields1D.free();
  free(nrs->coupling_flux);
  free(nrs->coupling_area);
  free(nrs->coupling_tmp1);
  free(nrs->coupling_tmp2);
  nrs->o_coupling_flux.free();
  nrs->o_coupling_area.free();
  nrs->o_coupling_tmp1.free();
  nrs->o_coupling_tmp2.free();
  nrs->o_coupling_cum.free();
  free(nrs->coupling_cum);
  nrs->o_fields1D_cum.free();
  

}
void copyFromNek(double time, int tstep) { nek::ocopyToNek(time, tstep); }

void udfExecuteStep(double time, int tstep, int isOutputStep)
{
  platform->timer.tic("udfExecuteStep", 1);
  if (isOutputStep) {
    nek::ifoutfld(1);
    nrs->isOutputStep = 1;
  }

  if (udf.executeStep)
    udf.executeStep(nrs, time, tstep);

  nek::ifoutfld(0);
  nrs->isOutputStep = 0;
  platform->timer.toc("udfExecuteStep");
}

void nekUserchk(void) { nek::userchk(); }

double dt(int tstep)
{
  if (platform->options.compareArgs("VARIABLE DT", "TRUE")) {
    if (tstep == 1) {
      double initialDt = 0.0;
      platform->options.getArgs("DT", initialDt);
      if (initialDt > 0.0) {
        nrs->dt[0] = initialDt;
        return nrs->dt[0];
      }
    }
    const double dtOld = nrs->dt[0];
    timeStepper::adjustDt(nrs, tstep);

    double maxDt = 0;
    platform->options.getArgs("MAX DT", maxDt);
    if (maxDt > 0)
      nrs->dt[0] = std::min(nrs->dt[0], maxDt);
  }

  nrsCheck(nrs->dt[0] < 1e-10 || std::isnan(nrs->dt[0]) || std::isinf(nrs->dt[0]),
           platform->comm.mpiComm,
           EXIT_FAILURE,
           "Invalid time step size %.2e\n",
           nrs->dt[0]);

  // during a neknek simulation, sync dt across all ranks
  if (nrs->neknek) {
    MPI_Allreduce(MPI_IN_PLACE, &nrs->dt[0], 1, MPI_DFLOAT, MPI_MIN, platform->comm.mpiCommParent);
  }

  return nrs->dt[0];
}

double writeInterval(void)
{
  double val = -1;
  platform->options.getArgs("SOLUTION OUTPUT INTERVAL", val);
  return val;
}

int writeControlRunTime(void)
{
  return platform->options.compareArgs("SOLUTION OUTPUT CONTROL", "SIMULATIONTIME");
}

int outputStep(double time, int tStep)
{
  int outputStep = 0;
  if (writeControlRunTime()) {
    double val;
    platform->options.getArgs("START TIME", val);
    if (lastOutputTime == 0 && val > 0)
      lastOutputTime = val;
    outputStep = ((time - lastOutputTime) + 1e-10) > nekrs::writeInterval();
  }
  else {
    if (writeInterval() > 0)
      outputStep = (tStep % (int)writeInterval() == 0);
  }

  if (enforceOutputStep) {
    enforceOutputStep = 0;
    return 1;
  }
  return outputStep;
}

void outputStep(int val) { nrs->isOutputStep = val; }

// nek5000 names its dumps <prefix><case><fid>.f<NNNNN>. NNNNN is NOT the time step: it is
// a counter (nfld) held in a SAVEd local array of mfo_open_files (nek5000/core/prepost.f)
// that is bumped once per dump and zero-initialised in every new process. Nothing on disk
// is ever consulted, so a restarted run numbers from 1 again and overwrites the files of
// the run it is continuing.
//
// That counter is a non-external symbol inside the nek5000 library, so it cannot be set
// from here and nothing passed down influences the name. Instead, when a shift is active
// we never let nek write under the real name at all: the dump goes out under a temporary
// prefix and is then moved into the continued sequence, so the previous run's files are
// never opened for writing. Shifting happens only when [CASEDATA] restart = 1; each of the
// two sequences gets its own shift (see resolveCheckpointOffset below).
//
// Both sequences are shifted: the classic one (empty suffix -> <case>0.f*) and the restart
// checkpoints ("restart" suffix -> restart<case>0.f*). nek keys its counters by prefix and
// i_find_prefix() matches them by substring, so the two temporaries must never be a
// substring of one another -- keeping them the same length guarantees that.
static constexpr const char *chkSuffix = "restart";
static constexpr const char *tmpPrefixFld = "tmpfld"; // classic sequence   (empty suffix)
static constexpr const char *tmpPrefixChk = "tmpchk"; // checkpoint sequence (chkSuffix)

static std::string fldName(const std::string &base, int fid, int n)
{
  char buf[32];
  snprintf(buf, sizeof(buf), "%d.f%05d", fid, n);
  return base + buf;
}

// Move the dump nek just wrote under tmpPrefix into slot (count + writeStepOffset) of the
// real sequence, and rewrite that sequence's .nek5000 index over the shifted names.
static void shiftLastDump(const std::string &realPrefix,
                          const std::string &tmpPrefix,
                          int count,
                          int writeStepOffset)
{
  std::string casename;
  platform->options.getArgs("CASENAME", casename);

  const std::string realBase = realPrefix + casename;
  const std::string tmpBase = tmpPrefix + casename;
  const int shifted = count + writeStepOffset;

  // one file per nek output group (nfileo); with a single output file this runs once
  int nMoved = 0;
  for (int fid = 0;; ++fid) {
    const auto src = fldName(tmpBase, fid, count);
    if (!fs::exists(src))
      break;

    const auto dst = fldName(realBase, fid, shifted);
    std::error_code ec;
    fs::rename(src, dst, ec);
    if (ec) {
      std::cout << "WARNING: could not move " << src << " -> " << dst << " (" << ec.message()
                << "); dump left under its temporary name\n";
      return;
    }
    ++nMoved;
  }

  if (nMoved == 0) {
    std::cout << "WARNING: expected dump " << fldName(tmpBase, 0, count)
              << " not found; numbering NOT shifted\n";
    return;
  }

  // nek wrote the index as <tmpBase>.nek5000, pointing at the tmp names and carrying its
  // own un-shifted count. Rewrite it as <realBase>.nek5000 over the continued sequence.
  std::ifstream in(tmpBase + ".nek5000");
  if (!in)
    return;

  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) {
    const auto at = line.find(tmpBase);
    if (at != std::string::npos)
      line.replace(at, tmpBase.size(), realBase); // filetemplate: drop the tmp prefix
    if (line.find("numtimesteps") != std::string::npos)
      line = " numtimesteps: " + std::to_string(shifted);
    lines.push_back(line);
  }
  in.close();

  std::ofstream out(realBase + ".nek5000", std::ios::trunc);
  for (const auto &line : lines)
    out << line << "\n";
  out.close();

  std::error_code ec;
  fs::remove(tmpBase + ".nek5000", ec);
}

// The two sequences advance at unrelated rates -- the classic one on writeInterval, the
// checkpoints once per coupling window -- so a single offset cannot serve both. The classic
// shift is [CASEDATA] writeStepOffset. The checkpoint shift is not a parameter at all: it is
// the number in [GENERAL] startFrom, which by construction is the last valid checkpoint of
// the run being continued. Resolved once and cached, so the warning is printed at most once.
static int resolveCheckpointOffset()
{
  static int cached = -1;
  if (cached >= 0)
    return cached;

  std::string startFrom;
  platform->options.getArgs("RESTART FILE NAME", startFrom);

  // startFrom looks like "restart<case>0.f00021" -- take the trailing .f<NNNNN>
  const auto at = startFrom.rfind(".f");
  int n = 0;
  if (at != std::string::npos && sscanf(startFrom.c_str() + at + 2, "%d", &n) == 1 && n > 0) {
    cached = n;
  }
  else {
    cached = 0;
    if (platform->comm.mpiRank == 0)
      std::cout << "WARNING: restart = 1 but no checkpoint number could be read from [GENERAL] "
                   "startFrom (\""
                << startFrom
                << "\"); restart checkpoints will renumber from 1 and overwrite the previous run\n";
  }
  return cached;
}

// Table mapping every restart checkpoint on disk to the classic dump number and time it
// corresponds to, so a later run can read off the writeStepOffset / startTime to continue
// from any checkpoint -- not just the last one. Rewritten in full at every checkpoint;
// rows at or beyond the checkpoint being written describe a future this run has just
// superseded and are dropped, so the table always matches what is actually on disk.
static void writeRestartIndex(int checkpoint, int lastDump, double t, double dt, int tStep)
{
  std::string casename;
  platform->options.getArgs("CASENAME", casename);

  // Coupling_dir/restart, so nek's and MURPHY's restart indices sit side by side next to
  // HOW_TO_RESTART.txt; fall back to the run directory if preciceConfig was never read.
  const fs::path dir = (couplingDir.empty() ? fs::path(".") : fs::path(couplingDir)) / "restart";
  std::error_code dirEc;
  fs::create_directories(dir, dirEc);
  const auto path = (dir / "nek.restartinfo").string();

  struct Row {
    int chk;
    int dump;
    double time;
    double dt;
    int step;
  };
  std::vector<Row> rows;

  std::ifstream in(path);
  if (in) {
    for (std::string line; std::getline(in, line);) {
      if (line.empty() || line[0] == '#')
        continue;

      Row r{};
      // a table written before the dt column existed still parses, with dt left at 0
      const bool ok =
          sscanf(line.c_str(), "%d %d %lg %lg %d", &r.chk, &r.dump, &r.time, &r.dt, &r.step) == 5 ||
          sscanf(line.c_str(), "%d %d %lg %d", &r.chk, &r.dump, &r.time, &r.step) == 4;
      if (ok && r.chk < checkpoint)
        rows.push_back(r);
    }
    in.close();
  }
  rows.push_back({checkpoint, lastDump, t, dt, tStep});

  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    std::cout << "WARNING: could not write " << path << "\n";
    return;
  }

  char buf[256];
  out << "# nekRS restart index -- case '" << casename << "'\n"
      << "# Rewritten at every restart checkpoint. One row per " << chkSuffix << casename
      << "0.f<checkpoint> on disk.\n"
      << "#\n"
      << "# To continue from a given checkpoint, put in the .par:\n"
      << "#   [GENERAL]  startFrom = " << chkSuffix << casename << "0.f<checkpoint>\n"
      << "#   [CASEDATA] restart = 1 ; startTime = <time> ; writeStepOffset = <lastDump>\n"
      << "# dt is the step size nekRS was running with at that checkpoint.\n"
      << "#\n";
  snprintf(buf, sizeof(buf), "# %10s %11s %24s %24s %10s", "checkpoint", "lastDump", "time", "dt",
           "tStep");
  out << buf << "\n";

  for (const auto &r : rows) {
    char chkStr[16];
    snprintf(chkStr, sizeof(chkStr), "%05d", r.chk);
    snprintf(buf, sizeof(buf), "  %10s %11d %24.15e %24.15e %10d", chkStr, r.dump, r.time, r.dt,
             r.step);
    out << buf << "\n";
  }
}

void outfld(double time, int step, std::string suffix, int writeStepOffset, bool restart)
{
  std::string oldValue;
  platform->options.getArgs("CHECKPOINT OUTPUT MESH", oldValue);

  const bool isChk = (suffix == chkSuffix);
  const std::string tmpPrefix = isChk ? tmpPrefixChk : tmpPrefixFld;

  // Write the mesh into the first file of EACH sequence, not just the first file of the
  // process. nek5000 has its own "first file carries the mesh" rule, but it is keyed on a
  // prefx3 that mfo_open_files never resets: chcopy(prefx3,prefix,min(len(prefix),3)) copies
  // nothing when the prefix is empty, so once a "restart" dump has set prefx3 to "res" the
  // blank-prefix test `prefx3.eq.'   '.and.nfld.eq.1` can no longer fire. With two sequences
  // running, whichever one dumps second is then left without coordinates and reads as
  // garbage geometry in ParaView.
  bool &firstOfSequence = isChk ? firstChkDump : firstFldDump;
  if (firstOfSequence)
    platform->options.setArgs("CHECKPOINT OUTPUT MESH", "TRUE");

  if (platform->options.compareArgs("MOVING MESH", "TRUE"))
    platform->options.setArgs("CHECKPOINT OUTPUT MESH", "TRUE");

  if (restart && writeStepOffset <= 0 && firstOutfld && platform->comm.mpiRank == 0)
    std::cout << "WARNING: restart = 1 but writeStepOffset = " << writeStepOffset
              << "; classic dumps will renumber from 1 and overwrite the previous run\n";

  // each sequence carries its own shift; 0 means "write where nek would have written"
  const int fldOff = restart ? writeStepOffset : 0;
  const int chkOff = restart ? resolveCheckpointOffset() : 0;
  const int off = isChk ? chkOff : fldOff;
  const bool shift = (off > 0);

  writeFld(nrs, time, step, shift ? tmpPrefix : suffix);

  // mirror nek's per-prefix counter, whether or not we are shifting: the restart index
  // needs the classic dump number in a first run too.
  int &count = isChk ? nekChkCount : nekDumpCount;
  ++count;

  if (shift || isChk) {
    // nek's I/O ranks (pid0 of each output group) must have closed their files first
    MPI_Barrier(platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0) {
      if (shift)
        shiftLastDump(suffix, tmpPrefix, count, off);
      if (isChk)
        writeRestartIndex(nekChkCount + chkOff, nekDumpCount + fldOff, time, nrs->dt[0], step);
    }
  }

  lastOutputTime = time;
  firstOutfld = 0;
  firstOfSequence = false;

  platform->options.setArgs("CHECKPOINT OUTPUT MESH", oldValue);
}

void outfld(double time, int step, int writeStepOffset, bool restart) { outfld(time, step, "", writeStepOffset, restart); }

double endTime(void)
{
  double endTime = -1;
  platform->options.getArgs("END TIME", endTime);
  return endTime;
}

int numSteps(void)
{
  int numSteps = -1;
  platform->options.getArgs("NUMBER TIMESTEPS", numSteps);
  return numSteps;
}

void lastStep(int val) { nrs->lastStep = val; }

int lastStep(double time, int tstep, double elapsedTime)
{
  if (!platform->options.getArgs("STOP AT ELAPSED TIME").empty()) {
    double maxElaspedTime;
    platform->options.getArgs("STOP AT ELAPSED TIME", maxElaspedTime);
    if (elapsedTime > 60.0 * maxElaspedTime)
      nrs->lastStep = 1;
  }
  else if (endTime() >= 0) {
    const double eps = 1e-12;
    nrs->lastStep = fabs((time + nrs->dt[0]) - endTime()) < eps || (time + nrs->dt[0]) > endTime();
  }
  else {
    nrs->lastStep = (tstep == numSteps());
  }

  if (enforceLastStep)
    return 1;
  return nrs->lastStep;
}

void *nekPtr(const char *id) { return nek::ptr(id); }

void *nrsPtr(void) { return nrs; }

int finalize(void) { return nrsFinalize(nrs); }

int runTimeStatFreq()
{
  int freq = 500;
  platform->options.getArgs("RUNTIME STATISTICS FREQUENCY", freq);
  return freq;
}

int printInfoFreq()
{
  int freq = 1;
  platform->options.getArgs("PRINT INFO FREQUENCY", freq);
  return freq;
}

int updateFileCheckFreq()
{
  int freq = 20;
  platform->options.getArgs("UPDATE FILE CHECK FREQUENCY", freq);
  return freq;
}

void printRuntimeStatistics(int step) { platform->timer.printRunStat(step); }

void processUpdFile()
{
  char *rbuf = nullptr;
  long long int fsize = 0;
  const std::string updFile = "nekrs.upd";

  if (rank == 0) {
    if (fs::exists(updFile)) {
      FILE *f = fopen(updFile.c_str(), "r");
      fseek(f, 0, SEEK_END);
      fsize = ftell(f);
      fseek(f, 0, SEEK_SET);
      rbuf = new char[fsize];
      fread(rbuf, 1, fsize, f);
      fclose(f);
      remove(updFile.c_str());
    }
  }

  MPI_Bcast(&fsize, 1, MPI_LONG_LONG_INT, 0, comm);

  if (fsize) {
    exit(1);
    if (rank == 0)
      std::cout << "processing " << updFile << " ...\n";

    if (rank != 0)
      rbuf = new char[fsize];
    MPI_Bcast(rbuf, fsize, MPI_CHAR, 0, comm);
    std::stringstream is;
    is.write(rbuf, fsize);
    inipp::Ini ini;
    ini.parse(is, false);

    std::string end;
    ini.extract("", "end", end);
    if (end == "true") {
      enforceLastStep = 1;
      platform->options.setArgs("END TIME", "-1");
    }

    std::string checkpoint;
    ini.extract("", "checkpoint", checkpoint);
    if (checkpoint == "true")
      enforceOutputStep = 1;

    std::string endTime;
    ini.extract("general", "endtime", endTime);
    if (!endTime.empty()) {
      if (rank == 0)
        std::cout << "  set endTime = " << endTime << "\n";
      platform->options.setArgs("END TIME", endTime);
    }

    std::string numSteps;
    ini.extract("general", "numsteps", numSteps);
    if (!numSteps.empty()) {
      if (rank == 0)
        std::cout << "  set numSteps = " << numSteps << "\n";
      platform->options.setArgs("NUMBER TIMESTEPS", numSteps);
    }

    std::string writeInterval;
    ini.extract("general", "writeinterval", writeInterval);
    if (!writeInterval.empty()) {
      if (rank == 0)
        std::cout << "  set writeInterval = " << writeInterval << "\n";
      platform->options.setArgs("SOLUTION OUTPUT INTERVAL", writeInterval);
    }

    delete[] rbuf;
  }
}

void printInfo(double time, int tstep, bool printStepInfo, bool printVerboseInfo)
{
  timeStepper::printInfo(nrs, time, tstep, printStepInfo, printVerboseInfo);
}

void verboseInfo(bool enabled)
{
  platform->options.setArgs("VERBOSE SOLVER INFO", "FALSE");
  if (enabled)
    platform->options.setArgs("VERBOSE SOLVER INFO", "TRUE");
}

void updateTimer(const std::string &key, double time) { platform->timer.set(key, time); }

void resetTimer(const std::string &key) { platform->timer.reset(key); }

int exitValue() { return platform->exitValue; }

void initStep(double time, double dt, int tstep) { timeStepper::initStep(nrs, time, dt, tstep); }

bool runStep(std::function<bool(int)> convergenceCheck, int corrector)
{
  return timeStepper::runStep(nrs, convergenceCheck, corrector);
}

bool runStep(int corrector)
{

  auto _nrs = &nrs;
  auto _udf = &udf;

  std::function<bool(int)> convergenceCheck = [](int corrector) -> bool {
    if (udf.timeStepConverged)
      return udf.timeStepConverged(nrs, corrector);
    else
      return true;
  };

  return timeStepper::runStep(nrs, convergenceCheck, corrector);
}

double finishStep()
{
  timeStepper::finishStep(nrs);
  return nrs->timePrevious + nrs->dt[0];
}

bool stepConverged() { return nrs->timeStepConverged; }

void couplingRead (double dt) {
  nrs->coupling->Read(dt);
  nrs->o_coupling_data1.copyFrom(nrs->coupling->Get_data1());
  nrs->o_coupling_data2.copyFrom(nrs->coupling->Get_data2());
}

void couplingWrite() {
  const auto Nfields = 3;
  const int np = nrs->coupling->direct_mesh_size();
  const auto offset = np;

  int n_VPM_mesh;
  if (nrs->coupling->staggered()) n_VPM_mesh = 3;
  else n_VPM_mesh = 1;
  nrs->interpolator->eval(Nfields,   // evaluation of the field o_U at the previously defined points, stored in o_fields1D
    nrs->fieldOffset, 
    nrs->o_U, 
    n_VPM_mesh * np, 
    nrs->o_fields1D);

  nrs->interpolator->eval(1,   // evaluation of the field o_U at the previously defined points, stored in o_fields1D
    nrs->fieldOffset, 
    nrs->o_coupling_cum, 
    n_VPM_mesh * np, 
    nrs->o_fields1D_cum );

  std::vector<dfloat> U_eval(n_VPM_mesh * np * Nfields);
  nrs->o_fields1D.copyTo(U_eval.data());
  std::vector<dfloat> cum_eval(n_VPM_mesh * np);
  nrs->o_fields1D_cum.copyTo(cum_eval.data());
  std::vector<double> * direct_data = nrs->coupling->direct_data();
  std::vector<double> * direct_data_cum = nrs->coupling->direct_data_cum();
  // The direct mesh hands us every MURPHY vertex inside nekRS's *axis-aligned* bounding
  // box, but the mesh is a curved O-mesh, so many of those points lie outside it. findpts
  // marks them CODE_NOT_FOUND and interpolator->eval leaves their entry in U_eval/cum_eval
  // UNDEFINED. Copying that on shipped uninitialised memory to MURPHY every step. Send an
  // explicit zero instead, cum included, so MURPHY's "no NW data here" test is meaningful.
  // Same guard nekRS uses throughout src/plugins/lpm.cpp.
  const auto &fp_code = nrs->interpolator->data().code;
  long n_unfound = 0;
  if (nrs->coupling->staggered()) {
    for (int i = 0; i < np; i++) {
      const bool f0 = (fp_code[0 * np + i] != findpts::CODE_NOT_FOUND);
      const bool f1 = (fp_code[1 * np + i] != findpts::CODE_NOT_FOUND);
      const bool f2 = (fp_code[2 * np + i] != findpts::CODE_NOT_FOUND);
      n_unfound += (!f0) + (!f1) + (!f2);
      (*direct_data)[3 * i + 0] = f0 ? U_eval[i + 0 * np] : 0.0;
      (*direct_data)[3 * i + 1] = f1 ? U_eval[3 * np + i + 1 * np] : 0.0;
      (*direct_data)[3 * i + 2] = f2 ? U_eval[6 * np + i + 2 * np] : 0.0;
      (*direct_data_cum)[3 * i + 0] = f0 ? std::round(cum_eval[i + 0 * np]) : 0.0;
      (*direct_data_cum)[3 * i + 1] = f1 ? std::round(cum_eval[1 * np + i ]) : 0.0;
      (*direct_data_cum)[3 * i + 2] = f2 ? std::round(cum_eval[2 * np + i ]) : 0.0;
    }
  } else {
    for (int i = 0; i < np; i++) {
      const bool found = (fp_code[i] != findpts::CODE_NOT_FOUND);
      n_unfound += (!found);
      const double cval = found ? std::round(cum_eval[i]) : 0.0;
      (*direct_data)[3 * i + 0] = found ? U_eval[i + 0 * np] : 0.0;
      (*direct_data)[3 * i + 1] = found ? U_eval[i + 1 * np] : 0.0;
      (*direct_data)[3 * i + 2] = found ? U_eval[i + 2 * np] : 0.0;
      (*direct_data_cum)[3 * i + 0] = cval;
      (*direct_data_cum)[3 * i + 1] = cval;
      (*direct_data_cum)[3 * i + 2] = cval;
    }
  }
  {
    // audit what we actually put on the wire, and how many points were unfound
    long n_bad = 0; double worst = 0.0;
    for (int i = 0; i < 3 * np; i++) {
      const double v = (*direct_data)[i], c = (*direct_data_cum)[i];
      if (!std::isfinite(v) || std::fabs(v) > 1.0e10 || !std::isfinite(c) || std::fabs(c) > 1.0e10) {
        n_bad++; if (std::fabs(v) > std::fabs(worst)) worst = v;
      }
    }
    long n_tot = (long)(nrs->coupling->staggered() ? 3 : 1) * (long)np;
    long g_unf = 0, g_tot = 0, g_bad = 0;
    MPI_Reduce(&n_unfound, &g_unf, 1, MPI_LONG, MPI_SUM, 0, platform->comm.mpiComm);
    MPI_Reduce(&n_tot,     &g_tot, 1, MPI_LONG, MPI_SUM, 0, platform->comm.mpiComm);
    MPI_Reduce(&n_bad,     &g_bad, 1, MPI_LONG, MPI_SUM, 0, platform->comm.mpiComm);
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    {
      const auto *vtx = nrs->coupling->direct_vertices();
      for (int i = 0; i < np; i++)
        for (int d = 0; d < 3; d++) {
          const double q = (*vtx)[3 * i + d];
          if (q < lo[d]) lo[d] = q;
          if (q > hi[d]) hi[d] = q;
        }
      double glo[3], ghi[3];
      MPI_Reduce(lo, glo, 3, MPI_DOUBLE, MPI_MIN, 0, platform->comm.mpiComm);
      MPI_Reduce(hi, ghi, 3, MPI_DOUBLE, MPI_MAX, 0, platform->comm.mpiComm);
      for (int d = 0; d < 3; d++) { lo[d] = glo[d]; hi[d] = ghi[d]; }
    }
    static bool reported = false;
    if (platform->comm.mpiRank == 0 && !reported) {
      printf("coupling: vertices nekRS RECEIVED span x[%g,%g] y[%g,%g] z[%g,%g]\n",
             lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
      printf("coupling: %ld of %ld vertices NOT found by findpts (sent as zero); "
             "send-buffer bad entries = %ld\n", g_unf, g_tot, g_bad);
      fflush(stdout); reported = true;
    }
  }
  if (platform->comm.mpiRank == 0) {
    printf("Interpolating nek velocity onto murphy vertices, and writing to preCICE\n");
  }

  nrs->coupling->Write();
 }

void couplingAdvance(double dt) { nrs->coupling->Advance(dt); }

double couplingMaxTimeStep() { return nrs->coupling->GetMaxTimeStep(); }

double coupling_dt(double coupling_max_dt, double dt_solver, double tol_floor_dt) {
  double dt; //final dt decided by the coupling
  double quotient = coupling_max_dt / dt_solver;

  double floored_quotient = std::floor(quotient);
  int floored_ratio = (int) floored_quotient;
  double floored_dt = coupling_max_dt / floored_quotient;

  double ceiled_quotient = std::ceil(quotient);
  int ceiled_ratio = (int) ceiled_quotient;
  double ceiled_dt = coupling_max_dt / ceiled_quotient;

  if ( (floored_dt - dt_solver) / dt_solver < tol_floor_dt) { //if proposed dt is larger than solver dt but within 10% (tol_floor_dt), we keep the proposed dt
    dt = floored_dt;
  } else { // else, we take the smaller dt
    dt = ceiled_dt;
  }
  nrs->dt[0] = dt;
  return dt;
}

bool isCouplingOngoing() {return nrs->coupling->IsCouplingOngoing(); }

void readCouplingParameters(std::string *config_file, bool *periodic_dir, double * periodic_bounds, int *M_VPM, bool *staggered, double *time, int *writeStepOffset, bool *restart, bool *pre_simulation, int *pre_sim_expected_last_step) {
  platform->par->extract("casedata", "preciceConfig", *config_file);
  // the restart index lives next to precice-config.xml (Coupling_dir), alongside MURPHY's
  couplingDir = fs::path(*config_file).parent_path().string();
  platform->par->extract("casedata", "periodicX", periodic_dir[0]);
  platform->par->extract("casedata", "periodicY", periodic_dir[1]);
  platform->par->extract("casedata", "periodicZ", periodic_dir[2]);
  platform->par->extract("casedata", "periodicXmin", periodic_bounds[0]);
  platform->par->extract("casedata", "periodicXmax", periodic_bounds[1]);
  platform->par->extract("casedata", "periodicYmin", periodic_bounds[2]);
  platform->par->extract("casedata", "periodicYmax", periodic_bounds[3]);
  platform->par->extract("casedata", "periodicZmin", periodic_bounds[4]);
  platform->par->extract("casedata", "periodicZmax", periodic_bounds[5]);
  platform->par->extract("casedata", "M_VPM", M_VPM[0]);
  platform->par->extract("casedata", "staggered", staggered[0]);
  platform->par->extract("casedata", "restart", restart[0]);
  if (restart[0]) {//if restart s false, dont read the startTime and writeStepOffset, as they are not needed. If restart is true, read them from the par file
    platform->par->extract("casedata", "startTime", time[0]); //for the restart, the user has to set the startTime in the par file to the restart time
    platform->par->extract("casedata", "writeStepOffset", writeStepOffset[0]); //offset for the output step counter
  }
  platform->par->extract("casedata", "pre_simulation", pre_simulation[0]);
  if (pre_simulation[0]) {
    platform->par->extract("casedata", "pre_sim_expected_last_step", pre_sim_expected_last_step[0]);
  }
} 
}//namespace nekrs

int nrsFinalize(nrs_t *nrs)
{
  auto exitValue = nekrs::exitValue();
  if (platform->options.compareArgs("BUILD ONLY", "FALSE")) {
    if (nrs->uSolver)
      delete nrs->uSolver;
    if (nrs->vSolver)
      delete nrs->vSolver;
    if (nrs->wSolver)
      delete nrs->wSolver;
    if (nrs->uvwSolver)
      delete nrs->uvwSolver;
    if (nrs->pSolver)
      delete nrs->pSolver;
    for (int is; is < nrs->Nscalar; is++) {
      if (nrs->cds->solver[is])
        delete nrs->cds->solver[is];
    }
    if (nrs->cvode)
      delete nrs->cvode;
    if (nrs->meshSolver)
      delete nrs->meshSolver;

    hypreWrapper::finalize();
    hypreWrapperDevice::finalize();
    AMGXfinalize();
    nek::finalize();
  }

  if (platform->comm.mpiRank == 0)
    std::cout << "finished with exit code " << exitValue << std::endl;

  return exitValue;
}
