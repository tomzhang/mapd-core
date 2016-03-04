#ifndef RENDERING_GL_GLWINDOW_H_
#define RENDERING_GL_GLWINDOW_H_

// #include "Types.h"
// #include "Window.h"
// #include "Viewport.h"
// #include "resources/GLResourceTypes.h"
// #include <GL/glew.h>
// #include <string>
#include "../../Window.h"
#include "Resources/Types.h"

#include <GL/glew.h>

namespace Rendering {

using Objects::Viewport;
using Settings::WindowSettings;

namespace GL {

/**
 * An abstract interface for windowing toolkits.
 *
 * This class is intended to be overwritten per system windowing library
 * to initialize the windowing system and create system-specific windows.
 *
 */
// class GLWindowImpl {
//  public:
//   ~GLWindowImpl();

//  protected:
//   GLWindowImpl();

//   std::string getName() const;

//  private:
//   friend class GLWindow;
// };

class GLWindow : public Window {
 public:
  virtual ~GLWindow();

  virtual size_t getGpuId() const = 0;

  // void makeActiveOnCurrentThread() final;
  // void makeInactive() final;
  // bool isActiveOnCurrentThread() final;

  // GLRenderer& getRenderer();
  // void bindFramebuffer() final;

  // TODO(croot): what if we want to blit a color, depth and/or stencil buffer?
  void blitFramebuffer(const Resources::GLFramebufferShPtr& fbo,
                       GLenum attachment,
                       const Viewport& srcFboBounds,
                       const Viewport& dstFboBounds,
                       GLenum filter = GL_NEAREST);

  // TODO(croot): what if we want to blit a color, depth and/or stencil buffer?
  void blitFramebuffer(const Resources::GLFramebufferShPtr& fbo,
                       GLenum attachment,
                       int srcFboX,
                       int srcFboY,
                       int srcFboWidth,
                       int srcFboHeight,
                       int dstFboX,
                       int dstFboY,
                       int dstFboWidth,
                       int dstFboHeight,
                       GLenum filter = GL_NEAREST);

  // std::shared_ptr<unsigned char> readPixelsRGBA(size_t startx, size_t starty, size_t width, size_t height) final;

 protected:
  explicit GLWindow(const WindowSettings& windowSettings, const WindowShPtr& prnt = nullptr);
  // explicit GLWindow(const WindowSettings& windowSettings,
  //                   const RendererShPtr& rendererPtr,
  //                   const WindowShPtr& prnt = nullptr);
  // explicit GLWindow(const WindowShPtr& prnt = nullptr);

  // GLRendererUqPtr _glRendererPtr;

  // virtual void _initRenderer() = 0;

  // void _initGLEW();

 private:
  // std::unique_ptr<GLWindowImpl> _implPtr;
  // bool _glewInitialized;
};

}  // namespace GL

}  // namespace Rendering

#endif  // RENDERING_GL_GLWINDOW_H_
