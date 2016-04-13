#ifndef RENDERING_WINDOW_H_
#define RENDERING_WINDOW_H_

#include "Types.h"
#include "Settings/WindowSettings.h"
#include "Objects/Viewport.h"
#include <string>
#include <vector>

namespace Rendering {

class Window {
 public:
  typedef std::vector<Renderer*> RendererArray;

  virtual ~Window();

  /** Returns true if the window has been properly initialized **/
  bool isInitialized() { return _initialized; }

  /**
   * Gets the settings of the window
   */
  Settings::WindowSettings getWindowSettings() const;

  /**
   * Gets the settings of the window as a constant reference.
   */
  const Settings::WindowSettings& getWindowSettingsRef() const;

  /**
   * Gets specific window settings
   */
  int64_t getIntSetting(const std::string& setting) const;
  int64_t getIntSetting(Settings::IntSetting setting) const;

  Settings::IntConstant getIntSettingAsConstant(const std::string& setting) const;
  Settings::IntConstant getIntSettingAsConstant(Settings::IntSetting setting) const;

  std::string getStrSetting(const std::string& setting) const;
  std::string getStrSetting(Settings::StrSetting setting) const;

  /**
   * Makes this window and its rendering context active on
   * the current thread. Note, only one context can be active
   * on a thread at a time, and a context can only be active on
   * one thread on a time. If this context is currently active
   * on a separate thread when this function is called,
   * then this context will be inactivated on that thread and
   * activated on the current thread. The other thread will result
   * in no context being active.
   */
  // perhaps this should only be on the renderer?
  // virtual void makeActiveOnCurrentThread() = 0;

  /**
   * This results in no context being active on the
   * current thread.
   */
  // virtual void makeInactive() = 0;

  /**
   * Returns true if this window and rendering context are
   * set to be current on the current thread.
   */
  // virtual bool isActiveOnCurrentThread() = 0;

  // virtual void bindFramebuffer() = 0;
  // virtual void bindDrawFramebuffer() = 0;

  // virtual const Framebuffer* getFramebuffer() const { return nullptr; }
  // virtual Framebuffer* getFramebuffer() { return nullptr; }

  // virtual std::shared_ptr<unsigned char> readPixelsRGB(size_t startx, size_t starty, size_t width, size_t height) =
  // 0;
  // virtual std::shared_ptr<unsigned char> readPixelsRGBA(size_t startx, size_t starty, size_t width, size_t height) =
  // 0;

  virtual void swapBuffers() = 0;

  // RendererShPtr getRenderer() const;

  Renderer* getPrimaryRenderer();
  RendererArray getAllRenderers();
  int numRenderers() const;

  // virtual void flush() = 0;
  // virtual void finish() = 0;

  /**
   * Viewport related funcs
   */

  /**
   * Sets the viewport for the window.
   * NOTE: The non-virtual function calls the virtual one
   * under the hood. Any subclass, if it needs to override
   * the virtual function, must call this function as well
   * to ensure full completion of setting the viewport.
   */
  void setViewport(const Objects::Viewport& viewport);
  void setViewport(int x, int y, int w, int h);

  /**
   * Gets a copy of the current viewport
   */
  Objects::Viewport getViewport() const;

  int getWidth() const;
  int getHeight() const;

  // void getColorFormat(); ?

  // TODO(croot): add an window type enum and get the window type?
  bool isGLWindow() { return _type == RendererType::OPENGL; }

  std::string getName() const;

  WindowShPtr getParent() const;

 protected:
  explicit Window(RendererType type, const Settings::WindowSettings& windowSettings, const WindowShPtr& prnt = nullptr);
  // explicit Window(const Settings::WindowSettings& windowSettings,
  //                 const RendererShPtr& rendererPtr,
  //                 const WindowShPtr& prnt = nullptr);
  // explicit Window(const WindowShPtr& prnt = nullptr);

  bool _initialized;
  RendererType _type;
  Settings::WindowSettings _settings;
  WindowShPtr _prnt;

  RendererArray _attachedRenderers;

  void attachRenderer(Renderer* renderer);
  void detachRenderer(Renderer* renderer, bool calledByRenderer = false);

 private:
  /* Called immediately after construction to initialize all resources **/
  void _initialize(Renderer* renderer);

  /**
   * Should be overridden in derived classed to initialize resources.
   * This will be called inside the initialize() method.
   */
  virtual void _init(Renderer* renderer) = 0;

  friend class Renderer;
  friend class WindowManager;
  friend class WindowManagerImpl;
};

}  // namespace Rendering

#endif  // RENDERING_WINDOW_H_
