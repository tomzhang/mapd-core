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

  void resize(size_t width, size_t height);

 private:
  void _createTx(::Rendering::GL::GLResourceManagerShPtr& rsrcMgr,
                 size_t width,
                 size_t height,
                 size_t idx,
                 size_t numTextures);
  void _initialize(size_t width, size_t height, size_t numTextures);
  void _cleanupUnusedTxs();

  struct AccumTxContainer {
    ::Rendering::GL::Resources::GLTexture2dShPtr tx;
    std::chrono::milliseconds lastUsedTime;

    // textures needs to stay aligned with the
    // compositor's copies. This alignment is
    // done with this index, which is provided after
    // the texture is registered with the compositor
    size_t accumTxIdx;

    AccumTxContainer(const ::Rendering::GL::Resources::GLTexture2dShPtr& tx, size_t idx) : tx(tx), accumTxIdx(idx) {
      lastUsedTime = getCurrentTimeMS();
    }
  };

  static bool containerCmp(const AccumTxContainer* a, const AccumTxContainer* b);

  ::Rendering::GL::Resources::GLPixelBuffer2dShPtr clearPboPtr;
  ::Rendering::GL::Resources::GLVertexBufferShPtr rectvbo;
  ::Rendering::GL::Resources::GLVertexArrayShPtr vao;

  struct ChangeLastUsedTime {
    ChangeLastUsedTime() : new_time(getCurrentTimeMS()) {}
    void operator()(AccumTxContainer& container) { container.lastUsedTime = new_time; }

   private:
    std::chrono::milliseconds new_time;
  };

  typedef ::boost::multi_index_container<
      AccumTxContainer,
      ::boost::multi_index::indexed_by<::boost::multi_index::hashed_unique<
          ::boost::multi_index::
              member<AccumTxContainer, ::Rendering::GL::Resources::GLTexture2dShPtr, &AccumTxContainer::tx>>>>
      AccumTxMap;

  ::Rendering::GL::GLRendererWkPtr _rendererPtr;
  QueryRenderCompositorShPtr _compositorPtr;
  AccumTxMap _accumTxMap;
  std::multiset<const AccumTxContainer*, std::function<bool(const AccumTxContainer*, const AccumTxContainer*)>>
      _inactiveAccumTxQueue;
  std::mutex _poolMtx;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYACCUMTXPOOL_H_
