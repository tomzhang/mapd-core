#ifndef QUERYRENDERER_QUERYACCUMTXPOOL_H_
#define QUERYRENDERER_QUERYACCUMTXPOOL_H_

#include "Types.h"
#include "../Utils/Utils.h"
#include <Rendering/Renderer/GL/Types.h>
#include <Rendering/Renderer/GL/Resources/Types.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <set>
#include <mutex>

namespace QueryRenderer {

class QueryAccumTxPool {
 public:
  static const size_t maxTextures;

  QueryAccumTxPool(::Rendering::GL::GLRendererShPtr& renderer, QueryRenderCompositorShPtr compositorPtr);
  ~QueryAccumTxPool();

  ::Rendering::GL::Resources::GLTexture2dShPtr getInactiveAccumTx(size_t width, size_t height);
  std::vector<::Rendering::GL::Resources::GLTexture2dShPtr> getInactiveAccumTx(size_t width,
                                                                               size_t height,
                                                                               size_t numTextures);

  void setAccumTxInactive(::Rendering::GL::Resources::GLTexture2dShPtr& tx);
  void setAccumTxInactive(const std::vector<::Rendering::GL::Resources::GLTexture2dShPtr>& txs);

  ::Rendering::GL::Resources::GLTexture2dShPtr getInactiveExtentsTx(::Rendering::GL::Resources::GLTexture2dShPtr& tx,
                                                                    bool calcStdDev = false);
  ::Rendering::GL::Resources::GLTexture2dShPtr getInactiveExtentsTx(
      ::Rendering::GL::Resources::GLTexture2dArray* txArray,
      bool calcStdDev = false);
  void setExtentsTxInactive(::Rendering::GL::Resources::GLTexture2dShPtr& tx);

  void resize(size_t width, size_t height);

 private:
  void _createTx(::Rendering::GL::GLResourceManagerShPtr& rsrcMgr,
                 size_t width,
                 size_t height,
                 size_t idx,
                 size_t numTextures);
  void _initialize(size_t width, size_t height, size_t numTextures);
  void _cleanupUnusedTxs();
  void _runExtentsPass(::Rendering::GL::GLRendererShPtr& renderer,
                       ::Rendering::GL::Resources::GLTexture2dShPtr& extentTx,
                       ::Rendering::GL::Resources::GLTexture2d* tx,
                       ::Rendering::GL::Resources::GLTexture2dArray* txArray,
                       bool calcStdDev);
  ::Rendering::GL::Resources::GLTexture2dShPtr _getInactiveExtentsTxInternal(
      ::Rendering::GL::Resources::GLTexture2d* tx,
      ::Rendering::GL::Resources::GLTexture2dArray* txArray,
      bool calcStdDev);

  struct TxContainer {
    ::Rendering::GL::Resources::GLTexture2dShPtr tx;
    std::chrono::milliseconds lastUsedTime;

    TxContainer(const ::Rendering::GL::Resources::GLTexture2dShPtr& tx) : tx(tx) { lastUsedTime = getCurrentTimeMS(); }
    virtual ~TxContainer() {}
  };

  struct AccumTxContainer : public TxContainer {
    // textures needs to stay aligned with the
    // compositor's copies. This alignment is
    // done with this index, which is provided after
    // the texture is registered with the compositor
    size_t accumTxIdx;

    AccumTxContainer(const ::Rendering::GL::Resources::GLTexture2dShPtr& tx, size_t idx)
        : TxContainer(tx), accumTxIdx(idx) {
      lastUsedTime = getCurrentTimeMS();
    }
  };

  static bool containerCmp(const AccumTxContainer* a, const AccumTxContainer* b);

  // TODO(croot): make the extents texture a 1D texture
  ::Rendering::GL::Resources::GLPixelBuffer2dShPtr clearExtentsPboPtr;
  ::Rendering::GL::Resources::GLTexture2dShPtr extentsTexturePtr;
  ::Rendering::GL::Resources::GLShaderShPtr extentsShaderPtr;
  ::Rendering::GL::Resources::GLShaderShPtr stdDevShaderPtr;

  ::Rendering::GL::Resources::GLPixelBuffer2dShPtr clearPboPtr;
  ::Rendering::GL::Resources::GLVertexBufferShPtr rectvbo;
  ::Rendering::GL::Resources::GLVertexArrayShPtr vao;

  struct ChangeLastUsedTime {
    ChangeLastUsedTime() : new_time(getCurrentTimeMS()) {}
    void operator()(std::shared_ptr<TxContainer>& container) { container->lastUsedTime = new_time; }

   private:
    std::chrono::milliseconds new_time;
  };

  typedef ::boost::multi_index_container<
      std::shared_ptr<TxContainer>,
      ::boost::multi_index::indexed_by<::boost::multi_index::hashed_unique<
          ::boost::multi_index::member<TxContainer, ::Rendering::GL::Resources::GLTexture2dShPtr, &TxContainer::tx>>>>
      TxMap;

  ::Rendering::GL::GLRendererWkPtr _rendererPtr;
  QueryRenderCompositorShPtr _compositorPtr;
  TxMap _accumTxMap;
  std::multiset<const AccumTxContainer*, std::function<bool(const AccumTxContainer*, const AccumTxContainer*)>>
      _inactiveAccumTxQueue;

  TxMap _extentTxMap;
  std::list<::Rendering::GL::Resources::GLTexture2dShPtr> _inactiveExtentTxQueue;
  std::mutex _poolMtx;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYACCUMTXPOOL_H_
