#include "NvidiaKernel.h"

#include <glog/logging.h>


char *generatePTX(const char *ll, size_t size, const char *filename);

namespace {

void fill_options(CUjit_option options[], void* optionValues[]) {
  options[0] = CU_JIT_LOG_VERBOSE;
  int log_verbose = 1;
  optionValues[0] = reinterpret_cast<void*>(log_verbose);
}

}

GpuExecutionContext::GpuExecutionContext(const std::string& llir_module,
                                         const std::string& func_name,
                                         const std::string& lib_path)
  : module_(nullptr)
  , kernel_(nullptr) {
  const unsigned int num_options = 1;
  CUjit_option options[num_options];
  void* optionValues[num_options];
  fill_options(options, optionValues);
  checkCudaErrors(cuLinkCreate(num_options, options, optionValues, &link_state));
  if (!lib_path.empty()) {
    // How to create a static CUDA library:
    // 1. nvcc -std=c++11 -arch=sm_30 --device-link -c [list of .cu files]
    // 2. nvcc -std=c++11 -arch=sm_30 -lib [list of .o files generated by step 1] -o [library_name.a]
    checkCudaErrors(cuLinkAddFile(link_state, CU_JIT_INPUT_LIBRARY, lib_path.c_str(),
      num_options, options, optionValues));
  }
  ptx = generatePTX(llir_module.c_str(), llir_module.size(), nullptr);
  CHECK(ptx);
  checkCudaErrors(cuLinkAddData(link_state, CU_JIT_INPUT_PTX, static_cast<void*>(ptx), strlen(ptx) + 1,
    0, num_options, options, optionValues));
  void* cubin;
  size_t cubinSize;
  checkCudaErrors(cuLinkComplete(link_state, &cubin, &cubinSize));
  checkCudaErrors(cuModuleLoadDataEx(&module_, cubin, num_options, options, optionValues));
  CHECK(module_);
  checkCudaErrors(cuModuleGetFunction(&kernel_, module_, func_name.c_str()));
}

GpuExecutionContext::~GpuExecutionContext() {
  auto status = cuModuleUnload(module_);
  checkCudaErrors(status);
  checkCudaErrors(cuLinkDestroy(link_state));
  free(ptx);
}
