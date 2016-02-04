#ifndef RENDERING_GL_GLX_X11DISPLAYMANAGER_H_
#define RENDERING_GL_GLX_X11DISPLAYMANAGER_H_

#include "Types.h"
#include <X11/Xresource.h>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace Rendering {
namespace GL {
namespace GLX {

class X11DisplayManager {
 public:
  X11DisplayManager();
  ~X11DisplayManager();

  size_t getNumGpus(const std::string& displayName = "") const;

  DisplayScreenPair connectToDisplay(const std::string& displayName = "");
  DisplayScreenPair connectToDisplay(const std::string& displayName, int screenId);
  DisplayScreenPair connectToDisplay(int screenId);

 private:
  typedef std::pair<X11DisplayWkPtr, int> DisplayDefaultScreenPair;
  typedef std::unordered_map<std::string, DisplayDefaultScreenPair> OpenDisplayMap;

  DisplayScreenPair _connectToDisplay(const std::string& displayName, int screenId);

  OpenDisplayMap _openedDisplays;
  std::mutex _rsrcMutex;
};

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_GLX_X11DISPLAYMANAGER_H_
