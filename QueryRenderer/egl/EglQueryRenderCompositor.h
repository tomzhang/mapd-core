#ifndef QUERYRENDERER_EGL_EGLQUERYRENDERCOMPOSITOR_H_
#define QUERYRENDERER_EGL_EGLQUERYRENDERCOMPOSITOR_H_

// #define GL_GLEXT_PROTOTYPES  // for GL extensions. NOTE: not using glew here as it
// doesn't support the GLES extensions we're going to
// use for EGLImage siblings

#define EGL_EGLEXT_PROTOTYPES  // for EGL extensions

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <Rendering/Renderer/GL/egl/Types.h>

#include "../QueryRenderCompositor.h"
#include <memory>

namespace QueryRenderer {

namespace EGL {

struct EglImage {
  EGLImage img;
  ::Rendering::GL::EGL::EglDisplayShPtr displayPtr;

  EglImage(const ::Rendering::GL::EGL::EglDisplayShPtr& displayPtr, EGLContext ctx, GLuint glId);
  ~EglImage();
};

class EglQueryRenderCompositor : public QueryRenderCompositor {
 public:
  ~EglQueryRenderCompositor();

  void resize(size_t width, size_t height) final;

 private:
  EglQueryRenderCompositor(::Rendering::GL::GLRenderer* renderer,
                           size_t width,
                           size_t height,
                           size_t numSamples = 1,
                           bool doHitTest = false,
                           bool doDepthTest = false);

  ::Rendering::GL::Resources::GLTexture2dShPtr _createFboTexture2dImpl(::Rendering::GL::GLRenderer* renderer,
                                                                       FboColorBuffer texType) final;

  ::Rendering::GL::Resources::GLRenderbufferShPtr _createFboRenderbufferImpl(::Rendering::GL::GLRenderer* renderer,
                                                                             FboRenderBuffer rboType) final;

  std::unique_ptr<EglImage> _rgbaEglImgPtr;
  std::unique_ptr<EglImage> _idEglImgPtr;
  std::unique_ptr<EglImage> _depthEglImgPtr;
};

}  // namespace EGL

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_EGL_EGLQUERYRENDERCOMPOSITOR_H need for_
