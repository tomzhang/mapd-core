#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <vector>

template <typename T> T * copyFromGpu (const T *devPtr, const size_t numeElems);

template <typename T> T * copyToGpu (const std::vector <T> &data);

template <typename T> void allocateGpuMem(T * &devMem, const size_t numElems);

size_t getMultiProcessorCount();
void setGpu (int gpuId);

#endif
