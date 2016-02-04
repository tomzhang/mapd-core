#ifndef RENDERING_WINDOWMANAGER_H_
#define RENDERING_WINDOWMANAGER_H_

#include "Types.h"
#include "Settings/Types.h"
#include <string>
#include <thread>
#include <utility>

namespace Rendering {

/**
 * An abstract interface for windowing toolkits.
 *
 * This class is intended to be overwritten per system windowing library
 * to initialize the windowing system and create system-specific windows.
 *
 */
class BaseWindowManagerImpl {
 public:
  virtual ~BaseWindowManagerImpl() {}

 protected:
  BaseWindowManagerImpl() {}

  virtual std::string getName() const = 0;
  virtual size_t getNumGpus() const { return 0; }
  virtual size_t getNumGpus(const Settings::WindowSettings& settings) const { return 0; }

  virtual WindowShPtr createWindow(const Settings::WindowSettings& settings) = 0;
  virtual RendererShPtr createRenderer(const Settings::RendererSettings& settings) = 0;

 private:
  friend class WindowManager;
};

class WindowManager {
 public:
  WindowManager();
  ~WindowManager();

  size_t getNumGpus() const;
  size_t getNumGpus(const Settings::WindowSettings& settings) const;

  WindowShPtr createWindow(const Settings::WindowSettings& settings);
  RendererShPtr createRenderer(const Settings::RendererSettings& settings);
  RendererShPtr createRendererForWindow(const Settings::RendererSettings& settings, const WindowShPtr& windowPtr);
  WindowShPtr createWindowWithRenderer(const Settings::WindowSettings& settings, const RendererShPtr& rendererPtr);
  std::pair<WindowShPtr, RendererShPtr> createWindowAndRenderer(const Settings::WindowSettings& settings);

  void addRendererToWindow(const RendererShPtr& rendererPtr, const WindowShPtr& windowPtr);
  void addWindowToRenderer(const WindowShPtr& windowPtr, const RendererShPtr& rendererPtr);

 private:
  static const std::thread::id _mainThread;

  std::unique_ptr<BaseWindowManagerImpl> _implPtr;
};

}  // namespace Rendering

#endif  // RENDERING_WINDOWMANAGER_H_
