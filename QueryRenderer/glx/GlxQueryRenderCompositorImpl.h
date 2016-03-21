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
#include <unordered_set>

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

  void addFboTexture2d(::Rendering::GL::Resources::GLTexture2dShPtr& tex, FboColorBuffer texType) final;
  void addFboRenderbuffer(::Rendering::GL::Resources::GLRenderbufferShPtr& rbo, FboRenderBuffer rboType) final;

  void deleteFboTexture2d(::Rendering::GL::Resources::GLTexture2d* texture2dPtr) final;
  void deleteFboRenderbuffer(::Rendering::GL::Resources::GLRenderbuffer* renderbufferPtr) final;

  void render(QueryRenderer* queryRenderer) final;

 private:
  GlxQueryRenderCompositorImpl(QueryRenderManager* prnt,
                               ::Rendering::RendererShPtr& rendererPtr,
                               size_t width,
                               size_t height,
                               size_t numSamples = 1,
                               bool doHitTest = false,
                               bool doDepthTest = false);

  void _initResources(QueryRenderManager* queryRenderer);
  void _resizeImpl(size_t width, size_t height) final;

  ::Rendering::RendererWkPtr _rendererPtr;
  ::Rendering::GL::GLX::GlxGLRenderer* _renderer;
  ::Rendering::GL::Resources::GLVertexBufferShPtr _rectvbo;
  ::Rendering::GL::Resources::GLShaderShPtr _shader;
  ::Rendering::GL::Resources::GLVertexArrayShPtr _vao;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _rgbaTextureArray;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _idTextureArray;
  // ::Rendering::GL::Resources::GLTexture2dArrayShPtr _depthTextureArray;

  // std::unordered_map<::Rendering::GL::Resources::GLTexture2d*,
  //                    std::pair<::Rendering::GL::Resources::GLTexture2dWkPtr, int>> _rgbaTextures;
  // std::unordered_map<::Rendering::GL::Resources::GLTexture2d*,
  //                    std::pair<::Rendering::GL::Resources::GLTexture2dWkPtr, int>> _idTextures;
  // std::unordered_map<::Rendering::GL::Resources::GLRenderbuffer*,
  //                    std::pair<::Rendering::GL::Resources::GLRenderbufferWkPtr, int>> _rbos;

  std::unordered_set<::Rendering::GL::Resources::GLTexture2d*> _rgbaTextures;
  std::unordered_set<::Rendering::GL::Resources::GLTexture2d*> _idTextures;
  std::unordered_set<::Rendering::GL::Resources::GLRenderbuffer*> _rbos;

  friend class ::QueryRenderer::QueryRenderCompositor;
};

}  // namespace GLX
}  // namespace Impl
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_IMPL_GLX_GLXQUERYRENDERCOMPOSITORIMPL_H_
