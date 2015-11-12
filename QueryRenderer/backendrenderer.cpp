#include "QueryRenderManager.h"

// CROOT: CUDA COMMENT
#include "backendrendererSetup.h"
#include <cuda.h>

#include <glog/logging.h>

// #include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "gd.h"

using namespace MapD_Renderer;

int fbwWidthRatio, fbwHeightRatio;
int windowWidth, windowHeight;

QueryRenderManager renderManager(nullptr, 500000, nullptr, false);
// QueryRenderManager renderManager;

UserWidgetPair userWidgetId1 = std::make_pair(1, 1);
UserWidgetPair userWidgetId2 = std::make_pair(1, 2);
UserWidgetPair userWidgetId3 = std::make_pair(2, 1);

void mouse_btn_click_callback(GLFWwindow* window, int btn, int action, int mods) {
  double xpos, ypos;
  if (btn == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    glfwGetCursorPos(window, &xpos, &ypos);

    // invert y
    ypos = double(windowHeight) - ypos;

    // now compensate for a different-sized framebuffer (i.e. retina displays)
    xpos *= fbwWidthRatio;
    ypos *= fbwHeightRatio;
  }
}

void mouse_move_callback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos >= 0 && xpos <= windowWidth && ypos >= 0 && ypos <= windowHeight) {
    // std::cerr << "CROOT - in mouse move callback: " << xpos << ", " << ypos << std::endl;

    // invert y
    ypos = double(windowHeight) - ypos;

    // now compensate for a different-sized framebuffer (i.e. retina displays)
    xpos *= fbwWidthRatio;
    ypos *= fbwHeightRatio;

    unsigned int id = renderManager.getIdAt(xpos, ypos);
    std::cout << "CROOT - mouse move: (" << xpos << ", " << ypos << ") " << id << std::endl;
  }
}

// CROOT: CUDA COMMENT
template <typename T>
void checkCudaErrors(T result) {
  if (result) {
    fprintf(stderr, "CUDA error code=%d\n", static_cast<unsigned int>(result));
    cudaDeviceReset();
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

#define MAX(a, b) (a > b ? a : b)

inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {{0x20, 32},   // Fermi Generation (SM 2.0) GF100 class
                                     {0x21, 48},   // Fermi Generation (SM 2.1) GF10x class
                                     {0x30, 192},  // Kepler Generation (SM 3.0) GK10x class
                                     {0x32, 192},  // Kepler Generation (SM 3.2) GK10x class
                                     {0x35, 192},  // Kepler Generation (SM 3.5) GK11x class
                                     {0x37, 192},  // Kepler Generation (SM 3.7) GK21x class
                                     {0x50, 128},  // Maxwell Generation (SM 5.0) GM10x class
                                     {0x52, 128},  // Maxwell Generation (SM 5.2) GM20x class
                                     {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one to run properly
  printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n",
         major,
         minor,
         nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0, best_SM_arch = 0;
  int devices_prohibited = 0;

  unsigned long long max_compute_perf = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceCount(&device_count);

  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);

    // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
    if (deviceProp.computeMode != cudaComputeModeProhibited) {
      if (deviceProp.major > 0 && deviceProp.major < 9999) {
        best_SM_arch = MAX(best_SM_arch, deviceProp.major);
      }
    } else {
      devices_prohibited++;
    }

    current_device++;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  // Find the best CUDA capable GPU device
  current_device = 0;

  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);

    // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
    if (deviceProp.computeMode != cudaComputeModeProhibited) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
      }

      unsigned long long compute_perf =
          (unsigned long long)deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

      if (compute_perf > max_compute_perf) {
        // If we find GPU with SM major > 2, search only these
        if (best_SM_arch > 2) {
          // If our device==dest_SM_arch, choose this, or else pass
          if (deviceProp.major == best_SM_arch) {
            max_compute_perf = compute_perf;
            max_perf_device = current_device;
          }
        } else {
          max_compute_perf = compute_perf;
          max_perf_device = current_device;
        }
      }
    }

    ++current_device;
  }

  return max_perf_device;
}

typedef struct {
  int32_t dummy1;
  int32_t key;
  double x;
  double y;
  double val;
  int32_t dummy2;
  int32_t party;
} CROOTtest;

bool cudaInitialized = false;
QueryDataLayout runCuda(CudaHandle cudaHandle, int numPts = 100) {
  size_t sz = sizeof(Row);
  size_t fullSz = numPts * sz;

  CHECK(fullSz <= cudaHandle.numBytes);

  std::cout << "CROOT - bytes check - " << fullSz << "<" << cudaHandle.numBytes << std::endl;

  // CROOT: CUDA COMMENT
  curandState* d_states;
  cudaMalloc((void**)&d_states, sizeof(curandState) * numPts);
  setup_kernel(d_states, numPts, 1);

  cudaInitialized = true;

  cudaMemset(cudaHandle.handle, 0, numPts * sz);

  get_random_data(d_states, (Row*)cudaHandle.handle, numPts, numPts, 1);

  // CROOTtest* results = (CROOTtest*)calloc(numPts, sz);
  // cudaMemcpy(results, cudaHandle.handle, fullSz, cudaMemcpyDeviceToHost);
  // std::cout << "RESULTS: ";
  // for (int i = 0; i < numPts; ++i) {
  //   std::cout << " (" << results[i].dummy1 << " " << results[i].key << ", " << results[i].x << ", " << results[i].y
  //             << ", " << results[i].val << ", " << results[i].dummy2 << " " << results[i].party << ")";
  // }
  // std::cout << std::endl;
  // delete results;

  // std::vector<float> data = {0.145468,
  //                            0.434899,
  //                            0.87071,
  //                            2,
  //                            0.820181,
  //                            0.926417,
  //                            0.511765,
  //                            2,
  //                            0.550399,
  //                            0.811845,
  //                            0.78264,
  //                            3,
  //                            0.29483,
  //                            0.308556,
  //                            0.620706,
  //                            2,
  //                            0.914733,
  //                            0.557235,
  //                            0.554513,
  //                            1,
  //                            0.868979,
  //                            0.501246,
  //                            0.214082,
  //                            2,
  //                            0.321921,
  //                            0.206681,
  //                            0.118647,
  //                            1,
  //                            0.782857,
  //                            0.123377,
  //                            0.993959,
  //                            1,
  //                            0.0113023,
  //                            0.539587,
  //                            0.104572,
  //                            3,
  //                            0.28545,
  //                            0.198575,
  //                            0.231619,
  //                            1,
  //                            0.781606,
  //                            0.739071,
  //                            0.674268,
  //                            3,
  //                            0.23384,
  //                            0.877151,
  //                            0.138524,
  //                            1,
  //                            0.679064,
  //                            0.412877,
  //                            0.595212,
  //                            3,
  //                            0.282442,
  //                            0.918987,
  //                            0.815746,
  //                            2,
  //                            0.629903,
  //                            0.387381,
  //                            0.947378,
  //                            2,
  //                            0.121223,
  //                            0.225103,
  //                            0.224546,
  //                            3,
  //                            0.433255,
  //                            0.927185,
  //                            0.927222,
  //                            1,
  //                            0.383079,
  //                            0.945073,
  //                            0.0363172,
  //                            1,
  //                            0.513567,
  //                            0.8573,
  //                            0.941886,
  //                            1,
  //                            0.298722,
  //                            0.428918,
  //                            0.901925,
  //                            3,
  //                            0.416607,
  //                            0.570368,
  //                            0.502445,
  //                            3,
  //                            0.0344908,
  //                            0.549197,
  //                            0.169125,
  //                            2,
  //                            0.0493946,
  //                            0.740527,
  //                            0.783027,
  //                            3,
  //                            0.0466557,
  //                            0.2485,
  //                            0.882744,
  //                            1,
  //                            0.616587,
  //                            0.552467,
  //                            0.555897,
  //                            2};

  // glBindBuffer(GL_ARRAY_BUFFER, cudaHandle.handle);

  // // int numBytes = data.size() * sizeof(float) + (data.size() / 4) * sizeof(unsigned int);
  // // std::unique_ptr<char[]> byteDataPtr(new char[numBytes]);
  // // char* byteData = byteDataPtr.get();
  // // unsigned int id = 1;
  // // int startIdx = 0;
  // // for (int i = 0; i < data.size(); i += 4, ++id) {
  // //   memcpy(&byteData[startIdx], &data[i], 4 * sizeof(float));
  // //   startIdx += 4 * sizeof(float);
  // //   memcpy(&byteData[startIdx], &id, sizeof(unsigned int));
  // //   startIdx += sizeof(unsigned int);
  // // }
  // // glBufferSubData(GL_ARRAY_BUFFER, 0, numBytes, byteData);

  // int numBytes = data.size() * sizeof(float);
  // glBufferSubData(GL_ARRAY_BUFFER, 0, numBytes, &data[0]);

  // return Row::getQueryDataLayout(numPts);
  return ::MapD_Renderer::QueryDataLayout(numPts,
                                          {"key", "x", "y", "val", "party"},
                                          {::MapD_Renderer::QueryDataLayout::AttrType::INT64,
                                           ::MapD_Renderer::QueryDataLayout::AttrType::DOUBLE,
                                           ::MapD_Renderer::QueryDataLayout::AttrType::DOUBLE,
                                           ::MapD_Renderer::QueryDataLayout::AttrType::DOUBLE,
                                           ::MapD_Renderer::QueryDataLayout::AttrType::INT64});
}

int main(int argc, char* argv[]) {
  // GLFWwindow* window;

  CHECK(argc == 3);

  // setup google logging
  // google::InitGoogleLogging(argv[0]);
  // FLAGS_log_dir = "./LOGS";

  // CROOT: CUDA COMMENT
  CUcontext cudaCtx;
  int deviceId = gpuGetMaxGflopsDeviceId();
  cuCtxCreate(&cudaCtx, 0, deviceId);
  cuCtxSetCurrent(cudaCtx);

  std::string configJSON;

  std::ifstream inFile(argv[1]);
  inFile.seekg(0, std::ios::end);
  configJSON.reserve(inFile.tellg());
  inFile.seekg(0, std::ios::beg);
  configJSON.assign((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());

  renderManager.addUserWidget(userWidgetId1, false, false);
  renderManager.setActiveUserWidget(userWidgetId1);

  rapidjson::Document jsonDoc;
  jsonDoc.Parse(configJSON.c_str());
  if (jsonDoc.HasParseError()) {
    throw std::runtime_error("Error parsing the json config: offset: " + std::to_string(jsonDoc.GetErrorOffset()) +
                             ", error: " + rapidjson::GetParseError_En(jsonDoc.GetParseError()));
  }

  CHECK(jsonDoc.IsObject());

  rapidjson::Value::ConstMemberIterator mitr1, mitr2;
  rapidjson::Value::ConstValueIterator vitr;
  bool foundSql = false;

  if ((mitr1 = jsonDoc.FindMember("data")) != jsonDoc.MemberEnd() && mitr1->value.IsArray()) {
    for (vitr = mitr1->value.Begin(); vitr != mitr1->value.End(); ++vitr) {
      if (vitr->IsObject()) {
        if ((mitr2 = vitr->FindMember("sql")) != vitr->MemberEnd() && mitr2->value.IsString()) {
          foundSql = true;
        }
      }
    }
  }

  if (foundSql) {
    QueryDataLayout layout = runCuda(renderManager.getCudaHandle(), std::stoi(argv[2]));
    renderManager.configureRender(jsonDoc, &layout);
  } else {
    renderManager.configureRender(jsonDoc);
  }

  if (renderManager.inDebugMode()) {
    double lastTime = glfwGetTime();
    unsigned int nbFrames = 0;

    // Loop until the user closes the window
    GLFWwindow* window = renderManager.getWindow();

    // glfwSetMouseButtonCallback(window, mouse_btn_click_callback);
    glfwSetCursorPosCallback(window, mouse_move_callback);

    int framebufferWidth, framebufferHeight;

    // first need to get the true framebuffer size of the window
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

    fbwWidthRatio = float(framebufferWidth) / float(windowWidth);
    fbwHeightRatio = float(framebufferHeight) / float(windowHeight);

    while (!glfwWindowShouldClose(window)) {
      // Poll for and process events
      glfwPollEvents();

      renderManager.render();

      // Measure speed
      double currentTime = glfwGetTime();
      nbFrames++;
      if (currentTime - lastTime >= 1.0) {  // If last prinf() was more than 1 sec ago
        // printf and reset timer
        printf("%f ms/frame\n", 1000.0 / double(nbFrames));
        nbFrames = 0;
        lastTime += 1.0;
      }

      // Swap front and back buffers
      glfwSwapBuffers(window);
    }
  } else {
    PngData pngData = renderManager.renderToPng();
    pngData.writeToFile("out.png");

    std::cout << "DONE with render" << std::endl;
  }

  return 0;
}
