#if !defined(nekrs_nrs_hpp_)
#define nekrs_nrs_hpp_

#include <mpi.h>
#include <functional> 
#include <string>

namespace nekrs
{
void setup(MPI_Comm commg_in, MPI_Comm comm_in,
           int buildOnly, int commSizeTarget,
           int ciMode, std::string _setupFile,
           std::string _backend, std::string _deviceID,
           int _nSessions, int _sessionID,
           int debug);
void copyFromNek(double time, int tstep);
void udfExecuteStep(double time, int tstep, int isOutputStep);
void outfld(double time, int step);
void outfld(double time, int step, std::string suffix);
int outputStep(double time, int tStep);
void outputStep(int val);
int finalize();
void nekUserchk(void);
int runTimeStatFreq();
int printInfoFreq();
int updateFileCheckFreq();
void printRuntimeStatistics(int step);
double writeInterval(void);
double dt(int tStep);
double startTime(void);
double endTime(void);
int numSteps(void);
void lastStep(int val);
int lastStep(double time, int tstep, double elapsedTime);
int writeControlRunTime(void);
int exitValue(void);
bool stepConverged(void);
void processUpdFile();
void printInfo(double time, int tstep, bool printStepInfo, bool printVerboseInfo);
void verboseInfo(bool enabled);
void updateTimer(const std::string &key, double time);
void resetTimer(const std::string &key); 
void* nrsPtr(void);
void* nekPtr(const char* id);
void initStep(double time, double dt, int tstep);
bool runStep(std::function<bool(int)> convergenceCheck, int corrector);
bool runStep(int corrector);
double finishStep();
bool stepConverged();
void couplingRead (double dt);
void couplingWrite();
void couplingAdvance(double dt);
double couplingMaxTimeStep();
double coupling_dt(double coupling_max_dt, double dt_solver, double tol_floor_dt);
void couplingSetup(std::string_view config_file,std::string_view solver_name,
                   std::string_view mesh_name, std::string_view direct_mesh_name,
                   std::string_view data_name, std::string_view data2_name,
                   std::string_view direct_data_name, std::string_view direct_data_name_cum,
                   double tol_bb, bool *periodic_dir, double * periodic_bounds,
                   int M_VPM, bool staggered);
void couplingFinalize();
bool isCouplingOngoing();
void readCouplingParameters(std::string *config_file, bool *periodic_dir, double * periodic_bounds, int *M_VPM, bool *staggered);
}

#endif
