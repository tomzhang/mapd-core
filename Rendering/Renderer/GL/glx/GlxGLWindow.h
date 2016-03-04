#ifndef RENDERING_GL_GLX_GLXGLWINDOW_H_
#define RENDERING_GL_GLX_GLXGLWINDOW_H_

#include "Types.h"
#include <GL/glxew.h>
#include "../GLWindow.h"

namespace Rendering {
namespace GL {
namespace GLX {

typedef unsigned long DrawableID;

class GlxGLWindow : public GLWindow {
 public:
  ~GlxGLWindow();

  size_t getGpuId() const final;

  X11DisplayShPtr getXDisplayPtr() const;
  int getXScreenId() const;
  const DisplayScreenPair& getXConnectionData() const;

  // GLXContext getGLXContext() const;
  X11ID getXDrawable() const;

  // #ifdef GLEW_MX
  //   const GLXEWContext* glxewGetContext() const;
  //   GLXEWContext* glxewGetContext();
  // #endif

  // void makeActiveOnCurrentThread() final;
  // void makeInactive() final;

  void swapBuffers() final;

 private:
  GlxGLWindow(const Settings::WindowSettings& windowSettings,
              X11DisplayManager& displayMgr,
              const WindowShPtr& prnt = nullptr);

  /** The X11 Display Connection */
  DisplayScreenPair _dpyConnection;

  /** The X11 drawable ID (window or pbuffer) **/
  X11ID _xDrawable;

  /** The glX rendering context. */
  // GLXContext _glxContext;

  /** The renderer for this window. The GlxGLRenderer object holds the GLX context. */
  // GlxGLRendererUqPtr _renderer;

  void _setXDrawable(X11ID xDrawable);

  /**
   * Initialization functions. The _init() method overrides the virtual _init()
   * from the base Window class.
   */
  void _init(Renderer* renderer) final;
  void _initXDisplay(X11DisplayManager& displayMgr);
  FbConfigShPtr _chooseFbConfig(GlxGLRenderer* renderer);
  void _initRenderer(GlxGLRenderer* renderer);
  void _initXDrawable(GlxGLRenderer* renderer, const FbConfigShPtr& fbConfigPtr);

  void _initSwap(GlxGLRenderer* renderer);

  bool _configurePBuffer(GlxGLRenderer* renderer, const FbConfigShPtr& fbConfigPtr, X11ID& drawable);
  bool _configureWindow(GlxGLRenderer* renderer, const FbConfigShPtr& fbConfigPtr, X11ID& drawable);
  X11ID _createGlxWindow(GlxGLRenderer* renderer,
                         const FbConfigShPtr& fbConfigPtr,
                         int32_t x,
                         int32_t y,
                         int32_t w,
                         int32_t h);

  void _initEventHandler();
  void _exitEventHandler();

  friend class X11WindowManagerImpl;
};

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_GLX_GLXGLWINDOW_H_
