#ifndef QUERYRENDERER_QUERYIDMAPPBOPOOL_H_
#define QUERYRENDERER_QUERYIDMAPPBOPOOL_H_

#include "Types.h"
#include "../Utils/Utils.h"
#include "QueryIdMapPixelBuffer.h"
#include <Rendering/Renderer/GL/Types.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <list>
#include <mutex>

namespace QueryRenderer {

template <typename T = unsigned int>
class QueryIdMapPboPool {
 public:
  QueryIdMapPboPool(::Rendering::GL::GLRendererShPtr& renderer) : _rendererPtr(renderer) {}
  ~QueryIdMapPboPool() {}

  std::shared_ptr<QueryIdMapPixelBuffer<T>> getInactiveIdMapPbo(size_t width, size_t height) {
    std::lock_guard<std::mutex> pool_lock(_poolMtx);

    std::shared_ptr<QueryIdMapPixelBuffer<T>> rtn;

    if (_inactivePboQueue.size() == 0) {
      auto renderer = _rendererPtr.lock();
      CHECK(renderer);
      rtn.reset(new QueryIdMapPixelBuffer<T>(renderer.get(), width, height));
      _pboMap.emplace(rtn);
    } else {
      rtn = _inactivePboQueue.front();
      _inactivePboQueue.pop_front();
      rtn->resize(width, height);
    }

    return rtn;
  }

  void setIdMapPboInactive(std::shared_ptr<QueryIdMapPixelBuffer<T>>& pbo) {
    std::lock_guard<std::mutex> pool_lock(_poolMtx);

    auto itr = _pboMap.find(pbo);
    CHECK(itr != _pboMap.end());

    _pboMap.modify(itr, ChangeLastUsedTime());

    // delete pbos that have been inactive for a while
    static const std::chrono::milliseconds maxPboIdleTime = std::chrono::milliseconds(300000);  // 5 minutes
    std::chrono::milliseconds cutoffTime = getCurrentTimeMS() - maxPboIdleTime;

    while (_inactivePboQueue.begin() != _inactivePboQueue.end()) {
      itr = _pboMap.find(_inactivePboQueue.front());
      CHECK(itr != _pboMap.end());

      if (itr->lastUsedTime >= cutoffTime) {
        break;
      }

      _pboMap.erase(itr);
      _inactivePboQueue.pop_front();
    }

    _inactivePboQueue.push_back(pbo);
  }

 private:
  struct IdMapPboContainer {
    std::shared_ptr<QueryIdMapPixelBuffer<T>> pbo;
    std::chrono::milliseconds lastUsedTime;

    IdMapPboContainer(const std::shared_ptr<QueryIdMapPixelBuffer<T>>& pbo) : pbo(pbo) {
      lastUsedTime = getCurrentTimeMS();
    }
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
          ::boost::multi_index::
              member<IdMapPboContainer, std::shared_ptr<QueryIdMapPixelBuffer<T>>, &IdMapPboContainer::pbo>>>> IdPboMap;

  ::Rendering::GL::GLRendererWkPtr _rendererPtr;
  IdPboMap _pboMap;
  std::list<std::shared_ptr<QueryIdMapPixelBuffer<T>>> _inactivePboQueue;
  std::mutex _poolMtx;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYIDMAPPBOPOOL_H_
