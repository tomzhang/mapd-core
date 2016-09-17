#ifndef RENDERING_RENDERER_H_
#define RENDERING_RENDERER_H_

#include "Types.h"
#include "Objects/Types.h"
#include "Settings/RendererSettings.h"
#include <vector>

namespace Rendering {

using Settings::RendererSettings;

class Renderer {
 public:
  typedef std::vector<Window*> WindowArray;

  virtual ~Renderer();

  virtual void initialize() = 0;

  virtual void makeActiveOnCurrentThread(Window* window = nullptr) = 0;
  void makeActiveOnCurrentThread(const WindowShPtr& window) {
    if (window && !window) {
      makeActiveOnCurrentThread(window.get());
    } else {
      makeActiveOnCurrentThread();
    }
  }

  virtual void makeInactive() = 0;
  virtual bool isActiveOnCurrentThread(Window* window = nullptr) = 0;

  bool isActiveOnCurrentThread(const WindowShPtr& window) {
    if (window) {
      return isActiveOnCurrentThread(window.get());
    } else {
      return isActiveOnCurrentThread();
    }
  }

  virtual void setClearColor(const Objects::ColorRGBA& color) = 0;
  virtual void setClearColor(float r, float g, float b, float a) = 0;

  virtual void clearColor() = 0;
  virtual void clearAll() = 0;

  virtual bool isGLRenderer() { return false; }

  /** Returns true if the renderer has been properly initialized **/
  bool isInitialized() { return _initialized; }

  Window* getPrimaryWindow();
  const Window* getPrimaryWindow() const;
  WindowArray getAllWindows();
  int numWindows() const;
  RendererId getId() const { return _id; }

 protected:
  Renderer(const RendererSettings& settings);

  /** A back references to the windows that this renderer can draw to **/
  WindowArray _attachedWindows;

  /** Set to true when the renderer has been fully initialized **/
  bool _initialized;

  /** Set to true when the _attachedWindows struct has changed **/
  /** It is up to derived classes to clear this flag when it is no longer needed **/
  bool _windowsDirty;

  RendererSettings _settings;

  /** Called by derived classes to set as initialized **/
  void setInitialized() { _initialized = true; }

  /**
   * Called to attach a window to the renderer. This function
   * should only be called by a Window instance.
   */
  void attachWindow(Window* window);

  /**
   * Called to detach a window from the renderer. This function
   * should only be called by a Window instance.
   */
  void detachWindow(Window* window);

 private:
  static RendererId rendererCnt;

  /** Verifies that a window is attachable. This is called
   * during the attachWindow() method and should be overridden
   * by derived classes.
   */
  virtual bool _verifyWindowIsAttachable(const Window* window) = 0;

  RendererId _id;

  friend class Window;
};

}  // namespace Rendering

#endif  // RENDERING_RENDERER_H_
