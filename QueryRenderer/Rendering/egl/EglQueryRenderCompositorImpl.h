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
#include "../../Scales/Types.h"
#include <memory>
#include <map>

namespace QueryRenderer {
namespace Impl {
namespace EGL {

typedef std::unordered_map<::Rendering::GL::Resources::GLTexture2d*, ::Rendering::GL::Resources::GLTexture2dWkPtr>
    AccumulatorMap;
typedef std::vector<AccumulatorMap> AccumulatorArray;

class EglQueryRenderCompositorImpl : public QueryRenderCompositorImpl {
 public:
  ~EglQueryRenderCompositorImpl();

  ::Rendering::GL::Resources::GLTexture2dShPtr createFboTexture2d(::Rendering::GL::GLRenderer* renderer,
                                                                  FboColorBuffer texType) final;

  ::Rendering::GL::Resources::GLRenderbufferShPtr createFboRenderbuffer(::Rendering::GL::GLRenderer* renderer,
                                                                        FboRenderBuffer rboType) final;

  void registerAccumulatorTexture(::Rendering::GL::Resources::GLTexture2dShPtr& tex,
                                  size_t accumIdx,
                                  size_t numTexturesInArray) final;

  void unregisterAccumulatorTexture(const ::Rendering::GL::Resources::GLTexture2dShPtr& tex, size_t accumIdx) final;

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
    std::vector<::Rendering::GL::Resources::GLTexture2dWkPtr> id1ATextures;
    std::vector<::Rendering::GL::Resources::GLTexture2dWkPtr> id1BTextures;
    std::vector<::Rendering::GL::Resources::GLTexture2dWkPtr> id2Textures;
    std::vector<::Rendering::GL::Resources::GLRenderbufferWkPtr> rbos;
    AccumulatorArray registeredAccumTxts;
  };

  EglQueryRenderCompositorImpl(QueryRenderManager* prnt,
                               ::Rendering::RendererShPtr& rendererPtr,
                               const size_t width,
                               const size_t height,
                               const size_t numSamples,
                               const bool doHitTest,
                               const bool doDepthTest,
                               const bool supportsInt64);

  void _resizeImpl(size_t width, size_t height) final;
  void _initAccumResources(size_t width, size_t height, size_t depth);
  void _cleanupAccumResources();

  void _postPassPerGpuCB(::Rendering::GL::GLRenderer* renderer,
                         QueryFramebufferUqPtr& framebufferPtr,
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

  std::unique_ptr<EglImage> _rgbaEglImgPtr;
  std::unique_ptr<EglImage> _id1AEglImgPtr;
  std::unique_ptr<EglImage> _id1BEglImgPtr;
  std::unique_ptr<EglImage> _id2EglImgPtr;
  std::unique_ptr<EglImage> _depthEglImgPtr;

  std::map<GpuId, GLResourceStorage> _consumedRsrcs;

  std::vector<::Rendering::GL::Resources::GLTexture2dShPtr> _accumTexPtrArrayMap;
  std::vector<std::unique_ptr<EglImage>> _accumEglImgPtrArrayMap;

  friend class ::QueryRenderer::QueryRenderCompositor;
};

}  // namespace EGL
}  // namespace Impl
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_IMPL_EGL_EGLQUERYRENDERCOMPOSITORIMPL_H_
