#include "compileKernels.hpp"
#include "nrssys.hpp"
#include "platform.hpp"
#include "mesh.h" // for meshKernelProperties
#include "postProcessing.hpp"
void registerPostProcessingKernels()
{
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  const int Nq = N + 1;
  const int Np = Nq * Nq * Nq;

  auto kernelInfo = platform->kernelInfo + meshKernelProperties(N);

  kernelInfo["includes"].asArray();

  const std::string oklpath = getenv("NEKRS_KERNEL_DIR");
  std::string kernelName, fileName;

  // no atomics needed: registered ahead of the guard below, the coupled force dump uses it
  kernelName = "aeroForces";
  fileName = oklpath + "/postProcessing/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, kernelInfo);

  // pointwise, no atomics either: the coupled CFL / mesh Reynolds dump uses it
  kernelName = "cflMeshReynolds";
  fileName = oklpath + "/postProcessing/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, kernelInfo);

  // gatherPlanarValues and scatterPlanarValues kernels require use of atomics
  if (!platform->device.deviceAtomic)
    return;

  kernelName = "Qcriterion";
  fileName = oklpath + "/postProcessing/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, kernelInfo);

  kernelInfo["includes"] += oklpath + "/postProcessing/planarAveraging.h";

  for (const std::string dir : {"XY", "XZ", "YZ"}) {
    kernelName = "gatherPlanarValues" + dir;
    fileName = oklpath + "/postProcessing/" + kernelName + ".okl";
    platform->kernels.add(kernelName, fileName, kernelInfo);

    kernelName = "scatterPlanarValues" + dir;
    fileName = oklpath + "/postProcessing/" + kernelName + ".okl";
    platform->kernels.add(kernelName, fileName, kernelInfo);
  }
}
