#ifndef QUERYRENDERER_IMPL_GLX_GLXQUERYRENDERCOMPOSITORIMPL_H_
#define QUERYRENDERER_IMPL_GLX_GLXQUERYRENDERCOMPOSITORIMPL_H_

// #define GL_GLEXT_PROTOTYPES  // for GL extensions. NOTE: not using glew here as it
// doesn't support the GLES extensions we're going to
// use for EGLImage siblings

#include <GL/glxew.h>

#include "../QueryRenderCompositor.h"
#include "../../Scales/Types.h"
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

  void registerAccumulatorTexture(::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                  size_t accumIdx,
                                  size_t numTexturesInArray) final;

  void unregisterAccumulatorTexture(const ::Rendering::GL::Resources::GLTexture2dShPtr& tex, size_t accumIdx) final;
  void unregisterAllAccumulatorTextures() final;

  void render(QueryRenderer* queryRenderer, const std::set<GpuId>& usedGpus) final;

 private:
  static const int maxAccumColors;

  GlxQueryRenderCompositorImpl(QueryRenderManager* prnt,
                               ::Rendering::RendererShPtr& rendererPtr,
                               const size_t width,
                               const size_t height,
                               const size_t numSamples,
                               const bool doHitTest,
                               const bool doDepthTest,
                               const bool supportsInt64);

  void _initAccumResources(size_t width, size_t height, size_t depth);
  void _cleanupAccumResources();
  void _initResources(QueryRenderManager* queryRenderer, const bool supportsInt64);
  void _resizeImpl(size_t width, size_t height) final;

  void _postPassPerGpuCB(::Rendering::GL::GLRenderer* renderer,
                         QueryFramebufferShPtr& framebufferPtr,
                         size_t width,
                         size_t height,
                         bool doHitTest,
                         bool doDepthTest,
                         int passCnt,
                         ScaleShPtr& accumulatorScalePtr,
                         int accumulatorCnt);

  void _compositePass(const std::set<GpuId>& usedGpus,
                      size_t width,
                      size_t height,
                      bool doHitTest,
                      bool doDepthTest,
                      int passCnt,
                      ScaleShPtr& accumulatorScalePtr);

  ::Rendering::RendererWkPtr _rendererPtr;
  ::Rendering::GL::GLX::GlxGLRenderer* _renderer;
  ::Rendering::GL::Resources::GLVertexBufferShPtr _rectvbo;
  ::Rendering::GL::Resources::GLShaderShPtr _shader;
  ::Rendering::GL::Resources::GLVertexArrayShPtr _vao;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _rgbaTextureArray;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _id1ATextureArray;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _id1BTextureArray;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _id2TextureArray;

  ::Rendering::GL::Resources::GLPixelBuffer2dShPtr _clearPboPtr;
  ::Rendering::GL::Resources::GLShaderShPtr _accumulatorShader;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _accumulationCpTextureArray;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _accumulationTextureArray;

  std::vector<
      std::unordered_map<::Rendering::GL::Resources::GLTexture2d*, ::Rendering::GL::Resources::GLTexture2dWkPtr>>
      _registeredAccumTxts;

  std::unordered_set<::Rendering::GL::Resources::GLTexture2d*> _rgbaTextures;
  std::unordered_set<::Rendering::GL::Resources::GLTexture2d*> _id1ATextures;
  std::unordered_set<::Rendering::GL::Resources::GLTexture2d*> _id1BTextures;
  std::unordered_set<::Rendering::GL::Resources::GLTexture2d*> _id2Textures;
  std::unordered_set<::Rendering::GL::Resources::GLRenderbuffer*> _rbos;

  friend class ::QueryRenderer::QueryRenderCompositor;
};

}  // namespace GLX
}  // namespace Impl
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_IMPL_GLX_GLXQUERYRENDERCOMPOSITORIMPL_H_
