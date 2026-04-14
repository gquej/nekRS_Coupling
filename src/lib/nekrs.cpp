#include <stdlib.h>
#include <filesystem>
#include <functional>
#include "nrs.hpp"
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
static bool startOfWindow = true;
static bool couplingWindowStateInitialized = false;
static double couplingWindowT0 = 0.0;
static double couplingWindowT1 = 0.0;
static std::vector<double> couplingData1Prev;
static std::vector<double> couplingData2Prev;
static std::vector<double> couplingData1Next;
static std::vector<double> couplingData2Next;

namespace nekrs {

void reset()
{
  lastOutputTime = 0;
  firstOutfld = 1;
  enforceLastStep = 0;
  enforceOutputStep = 0;
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
  std::string_view mesh_name, std::string_view direct_mesh_name,
  std::string_view data_name, std::string_view data2_name,
  std::string_view direct_data_name, std::string_view direct_data_name2, std::string_view direct_data_name3,
  std::string_view direct_data_name_cum,
  double tol_bb, bool *periodic_dir, double * periodic_bounds) {
  mesh_t *mesh = nrs->meshV;
  int p_Nfaces = mesh->Nfaces;
  int p_Nfp = mesh->Nfp;
  int p_Np = mesh->Np;
  int p_Nq = mesh->Nq;
  int N_elements = mesh->Nelements;

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
  nrs->coupling_bbox[1] = bbx2 + tol_bb; //bbx2 + tol_bb; Need to match Murphy especially for moving bodies
  nrs->coupling_bbox[2] = bby1 - tol_bb;
  nrs->coupling_bbox[3] = bby2 + tol_bb;
  nrs->coupling_bbox[4] = bbz1 - tol_bb;
  nrs->coupling_bbox[5] = bbz2 + tol_bb;

  platform->par->extract("casedata", "xmin", nrs->coupling_bbox[0]);
  platform->par->extract("casedata", "xmax", nrs->coupling_bbox[1]);
  platform->par->extract("casedata", "ymin", nrs->coupling_bbox[2]);
  platform->par->extract("casedata", "ymax", nrs->coupling_bbox[3]);
  platform->par->extract("casedata", "zmin", nrs->coupling_bbox[4]);
  platform->par->extract("casedata", "zmax", nrs->coupling_bbox[5]);
  nrs->coupling_bbox[0] -= tol_bb; //ensure we still have the tol_bb margin
  nrs->coupling_bbox[1] += tol_bb; //ensure we still have the tol_bb margin
  nrs->coupling_bbox[2] -= tol_bb; //ensure we still have the tol_bb margin
  nrs->coupling_bbox[3] += tol_bb; //ensure we still have the tol_bb margin
  nrs->coupling_bbox[4] -= tol_bb; //ensure we still have the tol_bb margin
  nrs->coupling_bbox[5] += tol_bb; //ensure we still have the tol_bb margin
  printf("Overriding coupling bbox xmax to %f from casedata\n", nrs->coupling_bbox[1]);

  nrs->coupling = new Coupling(solver_name, config_file);
  Coupling * coupling = nrs->coupling;

  int boundary_points_counter = 0;

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
      }
    }
  }
  double vertices_temp [3* boundary_points_counter]; //temp array to store all the outer vertices
  coupling->Resize_mapping(boundary_points_counter); //total number of nek outer vertices (as seen by nek)
  std::vector<int> * mapping = coupling->mapping(); //mapping from the nek mesh to the precice buffe
  int counter_to_idM[boundary_points_counter];
  
  //filling vertices_temp with all the outer vertices (including the doubles!!)

  int total_count = 0;

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
      }
    }
  }

  //filtering all the nek vertices that are duplicated across this rank (and only this rank)
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

  nrs->interpval1 = std::vector<double>(3 * unique_count, 0.0); // New vector for v at the unique vertices (for M2P)
  nrs->interpval2 = std::vector<double>(3 * unique_count, 0.0); // New vector for omega at the unique vertices (for M2P)

  nrs->o_coupling_data1 = platform->device.malloc(sizeof(double) * unique_count * 3, nrs->interpval1.data()); // This will have to change if M2P here
  nrs->o_coupling_data2 = platform->device.malloc(sizeof(double) * unique_count * 3, nrs->interpval2.data()); // Replaced coupling->Get_data1() with nrs->interpval1 for M2P
  nrs->o_coupling_mapping = platform->device.malloc(sizeof(int) * boundary_points_counter, coupling ->Get_mapping());

  nrs->o_coupling_data1.copyFrom(nrs->interpval1.data()); // Replaced coupling->Get_data1() with nrs->interpval1 for M2P
  nrs->o_coupling_data2.copyFrom(nrs->interpval2.data());
  nrs->o_coupling_mapping.copyFrom(coupling->Get_mapping());

  nrs->o_coupling_vmap = platform->device.malloc(mesh->Nelements * mesh->Np * sizeof(dlong), nrs->coupling_vmap);
  nrs->o_coupling_vmap.copyFrom(nrs->coupling_vmap);

  nrs->o_coupling_bbox = platform->device.malloc(6 * sizeof(dlong), nrs->coupling_bbox);
  nrs->o_coupling_bbox.copyFrom(nrs->coupling_bbox);
  coupling->Setup(mesh_name, direct_mesh_name, data_name, nrs->coupling_bbox, data2_name, direct_data_name, direct_data_name2, direct_data_name3, direct_data_name_cum);

  nrs->inverse_coupling_vmap.resize(unique_count, -1);
  for (int i = 0; i < unique_count; i++) {
    int idM = -1;
    for (int j = 0; j < mesh->Nelements * mesh->Np; j++) {
      if (nrs->coupling_vmap[j] == i) {
        nrs->inverse_coupling_vmap[i] = j;
        break;
      }
    }
  }
  //setting up the interpolator

  const int np = coupling->direct_mesh_size();
  const auto offset = np;
  nrs->interpolator = new pointInterpolation_t(nrs);
  nrs->o_fields1D = platform->device.malloc(3 * offset * sizeof(dfloat));
  std::vector<dfloat> xp, yp, zp;
  
  const std::vector<dfloat> *vertices = coupling->direct_vertices();

  for (int i = 0; i < np; i++) {
    xp.push_back((*vertices)[3 * i + 0]);
    yp.push_back((*vertices)[3 * i + 1]);
    zp.push_back((*vertices)[3 * i + 2]);
  }
  nrs->interpolator->setPoints(np, xp.data(), yp.data(), zp.data());
  nrs->interpolator->find();
  nrs->couplingMask.resize(np, 0);
  // Need to transform direct mesh in x, y, z vectors for M4'
  // Gather and filter coordinates and establish mapping to precice buffer :
  // Point at x[idx_x], y[idx_y], z[idx_z] -> data[3*i + 0/1/2] with i = idx_x + Nx * (idx_y + Ny * idx_z)
  
  std::vector<dfloat> x_VPM_mesh, y_VPM_mesh, z_VPM_mesh;
  x_VPM_mesh.reserve(np);
  y_VPM_mesh.reserve(np);
  z_VPM_mesh.reserve(np);

  dfloat x, y, z;
  for (int i = 0; i < np; i++) {
    x = (*vertices)[3 * i + 0];
    y = (*vertices)[3 * i + 1];
    z = (*vertices)[3 * i + 2];
    x_VPM_mesh.push_back(x);
    y_VPM_mesh.push_back(y);
    z_VPM_mesh.push_back(z);
  }

  auto unique_sorted = [tol](std::vector<dfloat> &values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end(), [tol](dfloat a, dfloat b) {
                  return fabs(a - b) < tol;
                }),
                values.end());
  };

  unique_sorted(x_VPM_mesh);
  unique_sorted(y_VPM_mesh);
  unique_sorted(z_VPM_mesh);

  auto findIndex = [tol](const std::vector<dfloat> &values, const dfloat value) -> int {
    auto it = std::lower_bound(values.begin(), values.end(), value);
    if (it != values.end() && fabs(*it - value) < tol) {
      return static_cast<int>(it - values.begin());
    }
    if (it != values.begin()) {
      auto prev = it - 1;
      if (fabs(*prev - value) < tol) {
        return static_cast<int>(prev - values.begin());
      }
    }
    return -1;
  };

  nrs->coupling_x_VPM = x_VPM_mesh;
  nrs->coupling_y_VPM = y_VPM_mesh;
  nrs->coupling_z_VPM = z_VPM_mesh;
  nrs->coupling_VPM_map.assign(x_VPM_mesh.size() * y_VPM_mesh.size() * z_VPM_mesh.size(), -1);

  const int Nx = x_VPM_mesh.size();
  const int Ny = y_VPM_mesh.size();
  for (int i = 0; i < np; ++i) {
    const dfloat xi = (*vertices)[3 * i + 0];
    const dfloat yi = (*vertices)[3 * i + 1];
    const dfloat zi = (*vertices)[3 * i + 2];

    const int idx_x = findIndex(x_VPM_mesh, xi);
    const int idx_y = findIndex(y_VPM_mesh, yi);
    const int idx_z = findIndex(z_VPM_mesh, zi);

    const int lin = idx_x + Nx * (idx_y + Ny * idx_z);
    nrs->coupling_VPM_map[lin] = i;
  }

  printf("VPM mapping: %d unique x, %d unique y, %d unique z\n", Nx, Ny, z_VPM_mesh.size());
  printf("VPM map size vs np : %d vs %d\n", nrs->coupling_VPM_map.size(), np);

}

void resetupInterpolator(){
  Coupling * coupling = nrs->coupling;
  const int np = coupling->direct_mesh_size();
  const auto offset = np;
  delete nrs->interpolator;
  nrs->o_fields1D.free();
  nrs->interpolator = new pointInterpolation_t(nrs);
  nrs->o_fields1D = platform->device.malloc(3 * offset * sizeof(dfloat));
  std::vector<dfloat> xp, yp, zp;
  
  const std::vector<dfloat> *vertices = coupling->direct_vertices();

  for (int i = 0; i < np; i++) {
    xp.push_back((*vertices)[3 * i + 0]);
    yp.push_back((*vertices)[3 * i + 1]);
    zp.push_back((*vertices)[3 * i + 2]);
  }
  nrs->interpolator->setPoints(np, xp.data(), yp.data(), zp.data());
  nrs->interpolator->find();
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
  mesh_t *mesh = nrs->meshV;
  mesh->o_U.copyTo(mesh->U);
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

void outfld(double time, int step, std::string suffix)
{
  std::string oldValue;
  platform->options.getArgs("CHECKPOINT OUTPUT MESH", oldValue);

  if (firstOutfld)
    platform->options.setArgs("CHECKPOINT OUTPUT MESH", "TRUE");

  if (platform->options.compareArgs("MOVING MESH", "TRUE"))
    platform->options.setArgs("CHECKPOINT OUTPUT MESH", "TRUE");

  writeFld(nrs, time, step, suffix);
  lastOutputTime = time;
  firstOutfld = 0;

  platform->options.setArgs("CHECKPOINT OUTPUT MESH", oldValue);
}

void outfld(double time, int step) { outfld(time, step, ""); }

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

double Weight(double i) {
  double dist = i;
  if (dist < 0.0) dist = -i;

  if (dist < 1.0) {
        return 1.0 + (-2.5 + 1.5 * dist) * dist * dist;
    } else if (dist < 2.0) {
        return 2.0 + (-4.0 + (2.5 - 0.5 * dist) * dist) * dist;
    } else {
        return 0.0;
    }
}

void M2Pinterp(nrs_t *nrs)
{
  Coupling *coupling = nrs->coupling;
  const std::vector<dfloat> &xVPM = nrs->coupling_x_VPM;
  const std::vector<dfloat> &yVPM = nrs->coupling_y_VPM;
  const std::vector<dfloat> &zVPM = nrs->coupling_z_VPM;
  const std::vector<int> &VPMmap = nrs->coupling_VPM_map;
  const dfloat xmin = xVPM[0];
  const dfloat ymin = yVPM[0];
  const dfloat zmin = zVPM[0];
  const dfloat hgrid = xVPM[1] - xVPM[0]; // Assuming uniform grid
  double *VPM_v = coupling->Get_data1(); // size: 3*np
  double *VPM_omega = coupling->Get_data2(); // size: 3*np
  const int Nx = xVPM.size();
  const int Ny = yVPM.size();
  const int Nz = zVPM.size();
  const int Nxy = Nx * Ny;

  const int size = coupling->mesh_size(); // Every vertex of Nek we target
  mesh_t *mesh = nrs->meshV;
  mesh->o_x.copyTo(mesh->x);
  mesh->o_y.copyTo(mesh->y);
  mesh->o_z.copyTo(mesh->z);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < size; i++) {
    const int idM = nrs->inverse_coupling_vmap[i];

    const double lx = (mesh->x[idM] - xmin) / hgrid;
    const double ly = (mesh->y[idM] - ymin) / hgrid;
    const double lz = (mesh->z[idM] - zmin) / hgrid;

    const int idx_x = (int) floor(lx);
    const int idx_y = (int) floor(ly);
    const int idx_z = (int) floor(lz);

    double acc1x = 0.0, acc1y = 0.0, acc1z = 0.0;
    double acc2x = 0.0, acc2y = 0.0, acc2z = 0.0;

    for (int i1 = -1; i1 < 3; i1++) {
      const double rx = lx - (double) (idx_x + i1);
      const double wx = Weight(rx);
      if (wx == 0.0)
        continue;

      int idx_xx = idx_x + i1;
      if (idx_xx < 0)
        idx_xx += Nx;
      else if (idx_xx >= Nx)
        idx_xx -= Nx;

      for (int j1 = -1; j1 < 3; j1++) {
        const double ry = ly - (double) (idx_y + j1);
        const double wy = Weight(ry);
        if (wy == 0.0)
          continue;

        int idx_yy = idx_y + j1;
        if (idx_yy < 0)
          idx_yy += Ny;
        else if (idx_yy >= Ny)
          idx_yy -= Ny;

        for (int k1 = -1; k1 < 3; k1++) {
          const double rz = lz - (double) (idx_z + k1);
          const double wz = Weight(rz);
          if (wz == 0.0)
            continue;

          const double w = wx * wy * wz;

          int idx_zz = idx_z + k1;
          if (idx_zz < 0)
            idx_zz += Nz;
          else if (idx_zz >= Nz)
            idx_zz -= Nz;

          const int lin = idx_xx + Nx * idx_yy + Nxy * idx_zz;
          const int map = 3 * VPMmap[lin];

          acc1x += w * VPM_v[map + 0];
          acc1y += w * VPM_v[map + 1];
          acc1z += w * VPM_v[map + 2];
          acc2x += w * VPM_omega[map + 0];
          acc2y += w * VPM_omega[map + 1];
          acc2z += w * VPM_omega[map + 2];
        }
      }
    }

    nrs->interpval1[3 * i + 0] += acc1x;
    nrs->interpval1[3 * i + 1] += acc1y;
    nrs->interpval1[3 * i + 2] += acc1z;
    nrs->interpval2[3 * i + 0] += acc2x;
    nrs->interpval2[3 * i + 1] += acc2y;
    nrs->interpval2[3 * i + 2] += acc2z;
  }
}

void couplingRead(double time, double dt, double coupling_max_dt, bool startOfCouplingWindow)
{
  Coupling *coupling = nrs->coupling;

  if (startOfCouplingWindow) {
    if (!couplingWindowStateInitialized) {
      coupling->Read(0.0);
      const int nData = 3 * coupling->direct_mesh_size();
      couplingData1Prev.resize(nData);
      couplingData2Prev.resize(nData);
      std::copy(coupling->Get_data1(), coupling->Get_data1() + nData, couplingData1Prev.begin());
      std::copy(coupling->Get_data2(), coupling->Get_data2() + nData, couplingData2Prev.begin());
      couplingData1Next.resize(nData);
      couplingData2Next.resize(nData);
      couplingWindowStateInitialized = true;
    }

    if (!couplingData1Next.empty()) {
      couplingData1Prev = couplingData1Next;
      couplingData2Prev = couplingData2Next;
    }

    coupling->Read(coupling_max_dt);
    const int nData = 3 * coupling->direct_mesh_size();
    std::copy(coupling->Get_data1(), coupling->Get_data1() + nData, couplingData1Next.begin());
    std::copy(coupling->Get_data2(), coupling->Get_data2() + nData, couplingData2Next.begin());

    couplingWindowT0 = time;
    couplingWindowT1 = time + coupling_max_dt;
  }

  const double denom = couplingWindowT1 - couplingWindowT0;
  double alpha = 0.0;
  if (denom > 0.0) {
    alpha = (time+dt - couplingWindowT0) / denom;
  }
  if (alpha < 0.0)
    alpha = 0.0;
  if (alpha > 1.0)
    alpha = 1.0;
  //printf("Time, alpha, T0, T1: %f, %f, %f, %f\n", time, alpha, couplingWindowT0, couplingWindowT1);
  const int nData = 3 * coupling->direct_mesh_size();
  double *VPM_v = coupling->Get_data1();
  double *VPM_omega = coupling->Get_data2();
  for (int i = 0; i < nData; ++i) {
    VPM_v[i] = (1.0 - alpha) * couplingData1Prev[i] + alpha * couplingData1Next[i];
    VPM_omega[i] = (1.0 - alpha) * couplingData2Prev[i] + alpha * couplingData2Next[i];
  }

  //printf("Reset values before M2P interpolation...\n");
  nrs->interpval1.assign(nrs->interpval1.size(), 0.0); // Reset interpolation values before M2P
  nrs->interpval2.assign(nrs->interpval2.size(), 0.0); // Reset interpolation values before M2P
  //printf("Read/interpolation done, starting M2P interpolation...\n");
  M2Pinterp(nrs);
  nrs->o_coupling_data1.copyFrom(nrs->interpval1.data()); // This will have to change if M2P here
  nrs->o_coupling_data2.copyFrom(nrs->interpval2.data()); // Replaced coupling->Get_data2() with nrs->interpval2 for M2P
  //printf("M2P interpolation done and data copied to device.\n");
}

double couplingWindowMeasurement(double coupling_max_dt) {
  double window_size;
  if (startOfWindow){
    startOfWindow = false;
    window_size = coupling_max_dt;
  } else {
    window_size = -1.0;
  }
  return window_size;
}

void couplingWrite(double dt_MURPHY) {
  const auto Nfields = 3;
  const int np = nrs->coupling->direct_mesh_size();
  const auto offset = np;
  resetupInterpolator();

  nrs->interpolator->eval(Nfields,   // evaluation of the field o_U at the previously defined points, stored in o_fields1D
    nrs->fieldOffset, 
    nrs->o_U, 
    offset, 
    nrs->o_fields1D);

  std::vector<dfloat> U_eval(Nfields * np);
  nrs->o_fields1D.copyTo(U_eval.data());
  std::vector<double> * direct_data = nrs->coupling->direct_data();
  std::vector<double> * direct_data_cum = nrs->coupling->direct_data_cum();
  std::vector<double> * direct_data2 = nrs->coupling->direct_data2();
  std::vector<double> * direct_data3 = nrs->coupling->direct_data3();
  std::vector<double> * global_data = nrs->coupling->global_data();
  std::vector<double> * global_data2 = nrs->coupling->global_data2();
  const std::vector<dfloat> *vertices = nrs->coupling->direct_vertices();
  dfloat xp, yp, zp, dist, vx, vy, vz;
  for (int i = 0; i < np; i++) {
    (*direct_data)[3 * i + 0] = U_eval[i + 0 * offset];
    (*direct_data)[3 * i + 1] = U_eval[i + 1 * offset];
    (*direct_data)[3 * i + 2] = U_eval[i + 2 * offset];
    xp = (*vertices)[3 * i + 0];
    yp = (*vertices)[3 * i + 1];
    zp = (*vertices)[3 * i + 2];
    // Hardcoded for cylinder body here (L = D)
    
    if (nrs->couplingMask[i] == 1) {
      vx = nrs->velocity[0] + nrs->omega[1]*(zp-nrs->position[2]) - nrs->omega[2]*(yp-nrs->position[1]);
      vy = nrs->velocity[1] + nrs->omega[2]*(xp-nrs->position[0]) - nrs->omega[0]*(zp-nrs->position[2]);
      vz = nrs->velocity[2] + nrs->omega[0]*(yp-nrs->position[1]) - nrs->omega[1]*(xp-nrs->position[0]);
      (*direct_data)[3 * i + 0] = vx;
      (*direct_data)[3 * i + 1] = vy; // Impose U = Ubody inside
      (*direct_data)[3 * i + 2] = vz;
    }
    (*direct_data_cum)[i] = 1.;
  }

  const int unique_count = nrs->coupling->mesh_size();
  mesh_t *mesh = nrs->meshV;
  // Ensure host buffer is up-to-date with device values before reading
  mesh->o_U.copyTo(mesh->U);

  // Use block layout: each field is stored in a block of size nrs->fieldOffset
  const dlong fieldOffset = nrs->fieldOffset;
  dfloat *Ux = mesh->U + 0 * fieldOffset;
  dfloat *Uy = mesh->U + 1 * fieldOffset;
  dfloat *Uz = mesh->U + 2 * fieldOffset;
  dfloat* coeffAB = mesh->coeffAB;
  int nAB = mesh->nAB;
  for (int i = 0; i < unique_count; i++) {
    int idM = -1;
    for (int j = 0; j < mesh->Nelements * mesh->Np; j++) {
      if (nrs->coupling_vmap[j] == i) {
        idM = j;
        break;
      }
    }
    if (idM != -1) {
      (*direct_data2)[3 * i + 0] = mesh->x[idM]; //+ dt_MURPHY * Ux[idM];
      (*direct_data2)[3 * i + 1] = mesh->y[idM]; //+ dt_MURPHY * Uy[idM];
      (*direct_data2)[3 * i + 2] = mesh->z[idM]; //+ dt_MURPHY * Uz[idM];
      (*direct_data3)[3 * i + 0] = Ux[idM];
      (*direct_data3)[3 * i + 1] = Uy[idM];
      (*direct_data3)[3 * i + 2] = Uz[idM];
      /*
      for (int k = 0; k < nAB; k++){
        dfloat coeff = coeffAB[k]/nrs->dt[0];
        (*direct_data3)[3 * i + 0] += Ux[idM + 3*k* fieldOffset] * coeff;
        (*direct_data3)[3 * i + 1] += Uy[idM + 3*k* fieldOffset] * coeff;
        (*direct_data3)[3 * i + 2] += Uz[idM + 3*k* fieldOffset] * coeff;
      }*/
    }
  }

  for (int i = 0; i < 3; i++){
    (*global_data)[i] = nrs->position[i];
    (*global_data)[3 + i] = nrs->velocity[i];
    (*global_data2)[i] = nrs->orientation[i];
    (*global_data2)[3 + i] = nrs->omega[i];
  }
  
  startOfWindow = true;
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

void readCouplingParameters(std::string *config_file, bool *periodic_dir, double * periodic_bounds) {
  platform->par->extract("casedata", "preciceConfig", *config_file);
  platform->par->extract("casedata", "periodicX", periodic_dir[0]);
  platform->par->extract("casedata", "periodicY", periodic_dir[1]);
  platform->par->extract("casedata", "periodicZ", periodic_dir[2]);
  platform->par->extract("casedata", "periodicXmin", periodic_bounds[0]);
  platform->par->extract("casedata", "periodicXmax", periodic_bounds[1]);
  platform->par->extract("casedata", "periodicYmin", periodic_bounds[2]);
  platform->par->extract("casedata", "periodicYmax", periodic_bounds[3]);
  platform->par->extract("casedata", "periodicZmin", periodic_bounds[4]);
  platform->par->extract("casedata", "periodicZmax", periodic_bounds[5]);
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
