#ifndef QUERYRENDERER_IMPL_GLX_GLXQUERYRENDERCOMPOSITORIMPL_H_
#define QUERYRENDERER_IMPL_GLX_GLXQUERYRENDERCOMPOSITORIMPL_H_

// #define GL_GLEXT_PROTOTYPES  // for GL extensions. NOTE: not using glew here as it
// doesn't support the GLES extensions we're going to
// use for EGLImage siblings

#include <GL/glxew.h>

#include "../QueryRenderCompositor.h"
#include <Rendering/Renderer/GL/glx/Types.h>
#include <Rendering/Renderer/GL/Resources/Types.h>
#include <memory>
#include <unordered_map>

namespace QueryRenderer {
namespace Impl {
namespace GLX {

class GlxQueryRenderCompositorImpl : public QueryRenderCompositorImpl {
 public:
  ~GlxQueryRenderCompositorImpl();

  ::Rendering::GL::Resources::GLTexture2dShPtr createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType) final;

  ::Rendering::GL::Resources::GLRenderbufferShPtr createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                        FboRenderBuffer rboType) final;

  void deleteFboTexture2d(const ::Rendering::GL::Resources::GLTexture2dShPtr& texture2dPtr) final;
  void deleteFboRenderbuffer(const ::Rendering::GL::Resources::GLRenderbufferShPtr& renderbufferPtr) final;

  void render(QueryRenderer* queryRenderer) final;

 private:
  GlxQueryRenderCompositorImpl(QueryRenderer* prnt,
                               ::Rendering::GL::GLRenderer* renderer,
                               size_t width,
                               size_t height,
                               size_t numSamples = 1,
                               bool doHitTest = false,
                               bool doDepthTest = false);

  void _resizeImpl(size_t width, size_t height) final;

  ::Rendering::GL::GLX::GlxGLRenderer* _renderer;

  // TODO(croot): Should we have a texture for each gpu, or can we get away with just one?
  std::unordered_map<::Rendering::GL::Resources::GLTexture2dShPtr, ::Rendering::GL::Resources::GLTexture2dShPtr>
      _rgbaTextures;
  std::unordered_map<::Rendering::GL::Resources::GLTexture2dShPtr, ::Rendering::GL::Resources::GLTexture2dShPtr>
      _idTextures;
  std::unordered_map<::Rendering::GL::Resources::GLRenderbufferShPtr, ::Rendering::GL::Resources::GLRenderbufferShPtr>
      _rbos;

  friend class ::QueryRenderer::QueryRenderCompositor;
};

}  // namespace GLX
}  // namespace Impl
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_IMPL_GLX_GLXQUERYRENDERCOMPOSITORIMPL_H_
