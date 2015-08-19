#include "gpu_utils.hpp"

#include <vector>

using namespace std;

size_t getMultiProcessorCount() {
  cudaDeviceProp cudaDevProps;
  cudaGetDeviceProperties(&cudaDevProps, 0);
  return cudaDevProps.multiProcessorCount;
}

void setGpu(int gpuId) {
  cudaSetDevice(gpuId);
}

template <typename T>
T* copyFromGpu(const T* devPtr, const size_t numElems) {
  T* hostPtr = new T[numElems];
  cudaMemcpy(hostPtr, devPtr, numElems * sizeof(T), cudaMemcpyDeviceToHost);
  return hostPtr;
}

template <typename T>
T* copyToGpu(const vector<T>& data) {
  size_t numElems = data.size();
  size_t elemSize = sizeof(T);
  size_t bufferSize = numElems * elemSize;
  const T* hostDataPtr = &data[0];
  T* deviceDataPtr;
  cudaMalloc((void**)&deviceDataPtr, bufferSize);
  cudaMemcpy(deviceDataPtr, hostDataPtr, bufferSize, cudaMemcpyHostToDevice);
  cudaError_t cudaError = cudaPeekAtLastError();
  if (cudaError != 0)
    return 0;
  // should clean up on error

  return deviceDataPtr;
}

template <typename T>
void allocateGpuMem(T*& devMem, const size_t numElems) {
  size_t elemSize = sizeof(T);
  cudaMalloc((void**)&devMem, numElems * elemSize);
  cudaMemset(devMem, 0, numElems * elemSize);
}

template void allocateGpuMem<char>(char*& devMem, const size_t numElems);
template void allocateGpuMem<int>(int*& devMem, const size_t numElems);
template void allocateGpuMem<unsigned int>(unsigned int*& devMem, const size_t numElems);
template void allocateGpuMem<float>(float*& devMem, const size_t numElems);
template void allocateGpuMem<double>(double*& devMem, const size_t numElems);

template unsigned int* copyFromGpu<unsigned int>(const unsigned int* devPtr, const size_t numeElems);
template int* copyFromGpu<int>(const int* devPtr, const size_t numeElems);

template char* copyToGpu<char>(const vector<char>& data);
template unsigned int* copyToGpu<unsigned int>(const vector<unsigned int>& data);
template int* copyToGpu<int>(const vector<int>& data);
template float* copyToGpu<float>(const vector<float>& data);
template double* copyToGpu<double>(const vector<double>& data);
