#ifndef RENDERING_GL_EGL_EGLDISPLAYMANAGER_H_
#define RENDERING_GL_EGL_EGLDISPLAYMANAGER_H_

#include "Types.h"
#include <EGL/egl.h>
#include <map>
#include <mutex>

namespace Rendering {
namespace GL {
namespace EGL {

class EglDisplay {
 public:
  EglDisplay(EGLDisplay& eglDpy);
  ~EglDisplay();

  EGLDisplay getEGLDisplay();

 private:
  EGLDisplay _dpy;
};

class EglDisplayManager {
 public:
  EglDisplayManager();
  ~EglDisplayManager();

  size_t getNumGpus() const;
  EglDisplayShPtr connectToDisplay(size_t deviceNum);

 private:
  typedef std::pair<int, int> MajorMinorPair;
  typedef std::pair<EglDisplayWkPtr, MajorMinorPair> DisplayVersionPair;
  typedef std::map<size_t, DisplayVersionPair> OpenDisplayMap;

  OpenDisplayMap _openedDisplayMap;

  std::mutex _rsrcMutex;
};

}  // namespace EGL
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_EGL_EGLDISPLAYMANAGER_H_
