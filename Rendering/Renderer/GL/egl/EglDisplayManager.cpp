#define EGL_EGLEXT_PROTOTYPES  // for EGL extensions
#include <EGL/egl.h>
#include <EGL/eglext.h>

#include "EglDisplayManager.h"
#include "MapDEGL.h"
#include "../../../RenderError.h"
#include <vector>
#include <algorithm>

namespace Rendering {
namespace GL {
namespace EGL {

struct EglDeviceInfo {
  EglDeviceInfo(size_t maxDevices) : eglDevices(maxDevices), numDevices(0) {
    static const PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");

    RUNTIME_EX_ASSERT(eglQueryDevicesEXT, "EGL_EXT_device_enumeration extension is not supported.");
    MAPD_CHECK_EGL_ERROR(eglQueryDevicesEXT(maxDevices, &eglDevices[0], reinterpret_cast<int*>(&numDevices)));

    static const PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT =
        (PFNEGLQUERYDEVICEATTRIBEXTPROC)eglGetProcAddress("eglQueryDeviceAttribEXT");
    RUNTIME_EX_ASSERT(eglQueryDeviceAttribEXT, "EGL_EXT_device_query extension is not supported.");

    std::sort(eglDevices.begin(), eglDevices.begin() + numDevices, [](EGLDeviceEXT a, EGLDeviceEXT b) {
      EGLAttrib aval, bval;

      eglQueryDeviceAttribEXT(a, EGL_CUDA_DEVICE_NV, &aval);
      eglQueryDeviceAttribEXT(b, EGL_CUDA_DEVICE_NV, &bval);

      return aval < bval;
    });
  }

  std::vector<EGLDeviceEXT> eglDevices;
  size_t numDevices;
};

static const EglDeviceInfo& getEglDeviceInfo() {
  static const EglDeviceInfo deviceInfo(32);  // NOTE: may need to increase this for systems with many gpus

  return deviceInfo;
}

static size_t getNumDevices() {
  return getEglDeviceInfo().numDevices;
}

static EGLDisplay getEGLDisplayFromDevice(size_t device) {
  static const PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

  const EglDeviceInfo info = getEglDeviceInfo();

  RUNTIME_EX_ASSERT(device < info.numDevices,
                    "Cannot get gpu device " + std::to_string(device) + ". There are only " +
                        std::to_string(info.numDevices) + " available.");
  return MAPD_CHECK_EGL_ERROR(eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, info.eglDevices[device], 0));
}

EglDisplay::EglDisplay(EGLDisplay& eglDpy) : _dpy(eglDpy) {
}

EglDisplay::~EglDisplay() {
  EGLBoolean hasErrors = MAPD_CHECK_EGL_ERROR(eglTerminate(_dpy));
  RUNTIME_EX_ASSERT(hasErrors == EGL_TRUE, "EGL error tyring to terminate EGL display.");
}

EGLDisplay EglDisplay::getEGLDisplay() {
  return _dpy;
}

static EglDisplayShPtr openEGLDisplay(size_t deviceNum, int& eglMajorVersion, int& eglMinorVersion) {
  EGLDisplay dpy = getEGLDisplayFromDevice(deviceNum);

  RUNTIME_EX_ASSERT(dpy != EGL_NO_DISPLAY, "Cannot open EGL display for device " + std::to_string(deviceNum) + ".");

  // now initialize the display
  EGLBoolean hasErrors = MAPD_CHECK_EGL_ERROR(eglInitialize(dpy, &eglMajorVersion, &eglMinorVersion));
  RUNTIME_EX_ASSERT(hasErrors == EGL_TRUE,
                    "EGL error trying to initialize display for device " + std::to_string(deviceNum));

  return EglDisplayShPtr(new EglDisplay(dpy));
}

EglDisplayManager::EglDisplayManager() {
}

EglDisplayManager::~EglDisplayManager() {
}

size_t EglDisplayManager::getNumGpus() const {
  return getNumDevices();
}

EglDisplayShPtr EglDisplayManager::connectToDisplay(size_t deviceNum) {
  EglDisplayShPtr rtnDisplayPtr;

  OpenDisplayMap::iterator itr;
  if ((itr = _openedDisplayMap.find(deviceNum)) == _openedDisplayMap.end()) {
    int eglMajorVersion, eglMinorVersion;
    rtnDisplayPtr = openEGLDisplay(deviceNum, eglMajorVersion, eglMinorVersion);

    _openedDisplayMap.insert(
        {deviceNum, std::make_pair(EglDisplayWkPtr(rtnDisplayPtr), std::make_pair(eglMajorVersion, eglMinorVersion))});
  } else {
    rtnDisplayPtr = itr->second.first.lock();
    if (!rtnDisplayPtr) {
      int eglMajorVersion, eglMinorVersion;
      rtnDisplayPtr = openEGLDisplay(deviceNum, eglMajorVersion, eglMinorVersion);
      itr->second.first = rtnDisplayPtr;
      itr->second.second.first = eglMajorVersion;
      itr->second.second.second = eglMinorVersion;
    }
  }

  return rtnDisplayPtr;
}

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering
