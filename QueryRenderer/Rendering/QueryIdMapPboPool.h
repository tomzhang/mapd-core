#ifndef QUERYRENDERER_QUERYIDMAPPBOPOOL_H_
#define QUERYRENDERER_QUERYIDMAPPBOPOOL_H_

#include "Types.h"
#include "../Utils/Utilities.h"
#include <Rendering/Renderer/GL/Types.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <list>
#include <mutex>

namespace QueryRenderer {

class QueryIdMapPboPool {
 public:
  QueryIdMapPboPool(::Rendering::GL::GLRendererShPtr& renderer);
  ~QueryIdMapPboPool();

  QueryIdMapPixelBufferShPtr getInactiveIdMapPbo(size_t width, size_t height);
  void setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo);

 private:
  struct IdMapPboContainer {
    QueryIdMapPixelBufferShPtr pbo;
    std::chrono::milliseconds lastUsedTime;

    IdMapPboContainer(const QueryIdMapPixelBufferShPtr& pbo) : pbo(pbo) { lastUsedTime = getCurrentTimeMS(); }
  };

  struct ChangeLastUsedTime {
    ChangeLastUsedTime() : new_time(getCurrentTimeMS()) {}
    void operator()(IdMapPboContainer& container) { container.lastUsedTime = new_time; }

   private:
    std::chrono::milliseconds new_time;
  };

  typedef ::boost::multi_index_container<
      IdMapPboContainer,
      ::boost::multi_index::indexed_by<::boost::multi_index::hashed_unique<
          ::boost::multi_index::member<IdMapPboContainer, QueryIdMapPixelBufferShPtr, &IdMapPboContainer::pbo>>>>
      IdPboMap;

  ::Rendering::GL::GLRendererWkPtr _rendererPtr;
  IdPboMap _pboMap;
  std::list<QueryIdMapPixelBufferShPtr> _inactivePboQueue;
  std::mutex _poolMtx;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYIDMAPPBOPOOL_H_
