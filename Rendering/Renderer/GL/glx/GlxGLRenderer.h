#ifndef RENDERING_GL_GLX_GLXGLRENDERER_H_
#define RENDERING_GL_GLX_GLXGLRENDERER_H_

#include "Types.h"
#include <GL/glxew.h>
#include "../GLRenderer.h"

namespace Rendering {
namespace GL {
namespace GLX {

class GlxGLRenderer : public GLRenderer {
 public:
  // GlxGLRenderer(const WindowShPtr& parentWindowPtr,
  //               const FbConfigUqPtr& fbConfigPtr);  // for full initialization at construction

  GlxGLRenderer(const RendererSettings& settings);
  // GlxGLRenderer(const WindowShPtr& parentWindowPtr);  // for delayed initialization
  ~GlxGLRenderer();

  GLXContext getGLXContext() const;
  X11DisplayShPtr getXDisplayPtr() const;
  int getXScreenId() const;

  /** for CROOT testing **/
  FbConfigShPtr getFbConfigPtr() const;

#ifdef GLEW_MX
  const GLXEWContext* glxewGetContext() const;
  GLXEWContext* glxewGetContext();
#endif

 private:
  /** The glX rendering context. */
  GLXContext _glxContext;

  /** The X11 Display Connection **/
  DisplayScreenPair _dpyConnection;

  /** CROOT - for testing **/
  FbConfigShPtr _fbConfigPtr;

  bool _glxewInitialized;

#ifdef GLEW_MX
  GLXEWContext _glxewContext;
#endif

  /** initialize function called by parent class **/
  void initializeGL() final;

  void _initXDisplay(const GlxGLWindow* primaryWindow);
  void _initGLX(const DisplayScreenPair& altDisplayData = DisplayScreenPair());
  FbConfigShPtr _chooseFbConfig(const GlxGLWindow* primaryWindow);

  bool _verifyWindowIsAttachable(const Window* window) final;

  void _makeActiveOnCurrentThreadImpl(Window* glWindow) final;
  void _makeInactiveImpl() final;

  friend class GlxGLWindow;
};

}  // namespace GLX
}  // namespace GL
}  // namespace Rendering

#endif  // RENDERING_GL_GLX_GLXGLRENDERER_H_
