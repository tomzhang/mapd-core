#ifndef QUERYRENDERER_IMPL_EGL_EGLQUERYRENDERCOMPOSITORIMPL_H_
#define QUERYRENDERER_IMPL_EGL_EGLQUERYRENDERCOMPOSITORIMPL_H_

// #define GL_GLEXT_PROTOTYPES  // for GL extensions. NOTE: not using glew here as it
// doesn't support the GLES extensions we're going to
// use for EGLImage siblings

#define EGL_EGLEXT_PROTOTYPES  // for EGL extensions

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <Rendering/Renderer/GL/egl/Types.h>
#include <CudaMgr/CudaMgr.h>

#include "../QueryRenderCompositor.h"
#include "../../Scales/Types.h"
#include <memory>
#include <map>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/random_access_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/tag.hpp>

#ifndef HAVE_CUDA
typedef int CUgraphicsResource;
#endif  // HAVE_CUDA

namespace QueryRenderer {
namespace Impl {
namespace EGL {

typedef std::unordered_map<::Rendering::GL::Resources::GLTexture2d*, ::Rendering::GL::Resources::GLTexture2dWkPtr>
    AccumulatorMap;
typedef std::vector<AccumulatorMap> AccumulatorArray;

class EglQueryRenderCompositorImpl : public QueryRenderCompositorImpl {
 public:
  struct CudaStateGuard {
    GpuId gpuId;
    CUcontext ctx;
    std::unordered_set<CUgraphicsResource> mappedCudaRsrcs;

    CudaStateGuard(EglQueryRenderCompositorImpl* currCompositor);
    CudaStateGuard(const CUcontext& ctx, const GpuId gpuId);
    ~CudaStateGuard();

    void mapCudaRsrc(CUgraphicsResource& cudaRsrc);
    void mapCudaRsrcs(std::vector<CUgraphicsResource>& cudaRsrcs);
    void unmapCudaRsrc(CUgraphicsResource& cudaRsrc);
    void unmapAllCudaRsrcs();
  };

  ~EglQueryRenderCompositorImpl();

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
  void _initResources(QueryRenderManager* queryRenderer, const bool supportsInt64);

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

  GpuId _startGpuId;
  const CudaMgr_Namespace::CudaMgr* _cudaMgr;

  ::Rendering::GL::GLRendererShPtr _rendererPtr;
  ::Rendering::GL::Resources::GLVertexBufferShPtr _rectvbo;
  ::Rendering::GL::Resources::GLShaderShPtr _shader;
  ::Rendering::GL::Resources::GLVertexArrayShPtr _vao;

  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _rgbaTextureArray;
  CUgraphicsResource _rgbaRsrc;

  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _id1ATextureArray;
  CUgraphicsResource _id1ARsrc;

  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _id1BTextureArray;
  CUgraphicsResource _id1BRsrc;

  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _id2TextureArray;
  CUgraphicsResource _id2Rsrc;

  ::Rendering::GL::Resources::GLPixelBuffer2dShPtr _clearPboPtr;
  ::Rendering::GL::Resources::GLShaderShPtr _accumulatorShader;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _accumulationTextureArray;
  ::Rendering::GL::Resources::GLTexture2dArrayShPtr _accumulationCpTextureArray;
  CUgraphicsResource _accumulationCpRsrc;

  struct ConsumedRsrc {
    ::Rendering::GL::Resources::GLTexture2d* txPtr;
    ::Rendering::GL::Resources::GLTexture2dWkPtr txWkPtr;
    CUgraphicsResource txCudaRsrc;

    ConsumedRsrc() : txPtr(nullptr), txWkPtr(), txCudaRsrc(0) {}
    ConsumedRsrc(::Rendering::GL::Resources::GLTexture2dShPtr& txShPtr) : txPtr(txShPtr.get()), txWkPtr(txShPtr) {}

    void set(::Rendering::GL::Resources::GLTexture2dShPtr& tx) {
      txPtr = tx.get();
      txWkPtr = tx;
    }

    void empty() {
      txPtr = nullptr;
      txWkPtr.reset();
      txCudaRsrc = 0;
    }

    bool isEmpty() { return (txPtr == nullptr); }
  };

  struct UpdateCudaRsrc {
    UpdateCudaRsrc(CUgraphicsResource& rsrc) : newCudaRsrc(rsrc) {}
    UpdateCudaRsrc(CUgraphicsResource&& rsrc) : newCudaRsrc(rsrc) {}
    void operator()(ConsumedRsrc& rsrc) { rsrc.txCudaRsrc = newCudaRsrc; }

   private:
    CUgraphicsResource newCudaRsrc;
  };

  struct inorder {};
  typedef ::boost::multi_index_container<
      ConsumedRsrc,
      ::boost::multi_index::indexed_by<
          // hashed on resource pointer
          ::boost::multi_index::hashed_unique<::boost::multi_index::member<ConsumedRsrc,
                                                                           ::Rendering::GL::Resources::GLTexture2d*,
                                                                           &ConsumedRsrc::txPtr>>,
          ::boost::multi_index::random_access<::boost::multi_index::tag<inorder>>>>
      ConsumedRsrcMap;

  struct ConsumedResources {
    ConsumedRsrcMap rgbaTextures;
    ConsumedRsrcMap id1ATextures;
    ConsumedRsrcMap id1BTextures;
    ConsumedRsrcMap id2Textures;
    // TODO(croot): support depth

    std::vector<ConsumedRsrc> accumTextures;
  };
  bool _rebuildCudaConsumedRsrcs;

  std::map<GpuId, ConsumedResources> _consumedRsrcs;

  CudaStateGuard* _rndrCudaStateGuard;

  friend class ::QueryRenderer::QueryRenderCompositor;
  friend struct ActiveRendererGuard;
};

}  // namespace EGL
}  // namespace Impl
}  // namespace QueryRenderer

#endif  // QUERYRENDERER_IMPL_EGL_EGLQUERYRENDERCOMPOSITORIMPL_H_
