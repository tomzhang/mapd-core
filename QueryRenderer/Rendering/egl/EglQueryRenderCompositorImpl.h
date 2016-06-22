#ifndef QUERYRENDERER_IMPL_EGL_EGLQUERYRENDERCOMPOSITORIMPL_H_
#define QUERYRENDERER_IMPL_EGL_EGLQUERYRENDERCOMPOSITORIMPL_H_

// #define GL_GLEXT_PROTOTYPES  // for GL extensions. NOTE: not using glew here as it
// doesn't support the GLES extensions we're going to
// use for EGLImage siblings

#define EGL_EGLEXT_PROTOTYPES  // for EGL extensions

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <Rendering/Renderer/GL/egl/Types.h>

#include "../QueryRenderCompositor.h"
#include <memory>
#include <map>

namespace QueryRenderer {
namespace Impl {
namespace EGL {

class EglQueryRenderCompositorImpl : public QueryRenderCompositorImpl {
 public:
  ~EglQueryRenderCompositorImpl();

  ::Rendering::GL::Resources::GLTexture2dShPtr createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType) final;

  ::Rendering::GL::Resources::GLRenderbufferShPtr createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                        FboRenderBuffer rboType) final;

  void render(QueryRenderer* queryRenderer, const std::set<GpuId>& usedGpus) final;

 private:
  struct EglImage {
    EGLImage img;
    ::Rendering::GL::EGL::EglDisplayShPtr displayPtr;

    EglImage(const ::Rendering::GL::EGL::EglDisplayShPtr& displayPtr, EGLContext ctx, GLuint glId);
    ~EglImage();
  };

  struct GLResourceStorage {
    std::vector<::Rendering::GL::Resources::GLTexture2dWkPtr> rgbaTextures;
    std::vector<::Rendering::GL::Resources::GLTexture2dWkPtr> idTextures;
    std::vector<::Rendering::GL::Resources::GLRenderbufferWkPtr> rbos;
  };

  EglQueryRenderCompositorImpl(QueryRenderManager* prnt,
                               ::Rendering::RendererShPtr& rendererPtr,
                               size_t width,
                               size_t height,
                               size_t numSamples = 1,
                               bool doHitTest = false,
                               bool doDepthTest = false);

  void _resizeImpl(size_t width, size_t height) final;

  std::unique_ptr<EglImage> _rgbaEglImgPtr;
  std::unique_ptr<EglImage> _idEglImgPtr;
  std::unique_ptr<EglImage> _depthEglImgPtr;

  std::map<GpuId, GLResourceStorage> _consumedRsrcs;

  friend class ::QueryRenderer::QueryRenderCompositor;
};

}  // namespace EGL
}  // namespace Impl
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_IMPL_EGL_EGLQUERYRENDERCOMPOSITORIMPL_H_
