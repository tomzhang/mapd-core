#ifndef QUERYRENDERER_BASEQUERYRSRCPOOL_H_
#define QUERYRENDERER_BASEQUERYRSRCPOOL_H_

#include "../Utils/Utils.h"
#include <Rendering/Renderer/GL/Types.h>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <list>
#include <mutex>

namespace QueryRenderer {

template <typename T, size_t maxInactiveTime, class... InitTypes>
class BaseQueryRsrcPool {
 public:
  BaseQueryRsrcPool(::Rendering::GL::GLRendererShPtr& renderer) : _rendererPtr(renderer) {}
  virtual ~BaseQueryRsrcPool() {}

  std::weak_ptr<T> getInactiveRsrc(InitTypes... args) {
    std::lock_guard<std::mutex> pool_lock(_poolMtx);

    std::shared_ptr<T> rtn;
    if (_inactiveRsrcQueue.size() == 0) {
      rtn = _initializeRsrc(args...);
      CHECK(rtn.use_count() == 1);
      _rsrcMap.emplace(rtn);
    } else {
      std::weak_ptr<T> rsrcWkPtr;
      while ((rsrcWkPtr = _inactiveRsrcQueue.front()).expired()) {
        _inactiveRsrcQueue.pop_front();
      }
      _inactiveRsrcQueue.pop_front();
      rtn = rsrcWkPtr.lock();
      CHECK(rtn);
      _updateRsrc(rtn, args...);
      CHECK(rsrcWkPtr.use_count() == 2);
    }

    _flushInactiveQueue();
    return rtn;
  }

  void setRsrcInactive(std::weak_ptr<T>& rsrcPtr) {
    std::lock_guard<std::mutex> pool_lock(_poolMtx);

    auto rsrcShPtr = rsrcPtr.lock();
    auto itr = _rsrcMap.find(rsrcShPtr);
    CHECK(itr != _rsrcMap.end() && rsrcPtr.use_count() == 2);

    _rsrcMap.modify(itr, ChangeLastUsedTime());
    _flushInactiveQueue();
    _inactiveRsrcQueue.push_back(rsrcPtr);
  }

  void deleteRsrc(std::weak_ptr<T>& rsrcPtr) {
    std::lock_guard<std::mutex> pool_lock(_poolMtx);
    {
      auto rsrcShPtr = rsrcPtr.lock();
      auto itr = _rsrcMap.find(rsrcShPtr);
      CHECK(itr != _rsrcMap.end() && rsrcPtr.use_count() == 2);
      _rsrcMap.erase(itr);
      _deleteRsrc(rsrcShPtr.get());
    }
    _flushInactiveQueue();
  }

 protected:
  virtual std::shared_ptr<T> _initializeRsrc(InitTypes... args) = 0;
  virtual void _updateRsrc(std::shared_ptr<T>& rsrcPtr, InitTypes... args) = 0;
  virtual void _deleteRsrc(const T* deletedRsrcPtr) {}
  ::Rendering::GL::GLRendererWkPtr _rendererPtr;

 private:
  void _flushInactiveQueue() {
    // delete rsrcs that have been inactive for a while
    static const std::chrono::milliseconds maxRsrcIdleTime = std::chrono::milliseconds(maxInactiveTime);
    std::chrono::milliseconds cutoffTime = getCurrentTimeMS() - maxRsrcIdleTime;

    while (_inactiveRsrcQueue.begin() != _inactiveRsrcQueue.end()) {
      auto ptr = _inactiveRsrcQueue.front();
      auto rsrcShPtr = ptr.lock();
      if (rsrcShPtr) {
        auto itr = _rsrcMap.find(rsrcShPtr);
        CHECK(itr != _rsrcMap.end());

        if (itr->lastUsedTime >= cutoffTime) {
          break;
        }

        _rsrcMap.erase(itr);
        _deleteRsrc(rsrcShPtr.get());
      }
      _inactiveRsrcQueue.pop_front();
    }
  }

  struct RsrcContainer {
    std::shared_ptr<T> rsrcPtr;
    std::chrono::milliseconds lastUsedTime;

    RsrcContainer(const std::shared_ptr<T>& rsrcPtr) : rsrcPtr(rsrcPtr) { lastUsedTime = getCurrentTimeMS(); }
  };

  struct ChangeLastUsedTime {
    ChangeLastUsedTime() : new_time(getCurrentTimeMS()) {}
    void operator()(RsrcContainer& container) { container.lastUsedTime = new_time; }

   private:
    std::chrono::milliseconds new_time;
  };

  typedef ::boost::multi_index_container<
      RsrcContainer,
      ::boost::multi_index::indexed_by<::boost::multi_index::hashed_unique<
          ::boost::multi_index::member<RsrcContainer, std::shared_ptr<T>, &RsrcContainer::rsrcPtr>>>>
      RsrcMap;

  RsrcMap _rsrcMap;
  std::list<std::weak_ptr<T>> _inactiveRsrcQueue;
  std::mutex _poolMtx;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_BASEQUERYRSRCPOOL_H_
