#ifndef QUERYRENDERER_QUERYBUFFERPOOL_H_
#define QUERYRENDERER_QUERYBUFFERPOOL_H_

#include "BaseQueryRsrcPool.h"

namespace QueryRenderer {

template <typename T>
class QueryBufferPool : public BaseQueryRsrcPool<T, 300000, size_t> {
 public:
  QueryBufferPool(::Rendering::GL::GLRendererShPtr& renderer)
      : BaseQueryRsrcPool<T, 300000, size_t>(renderer), _totalUsedBytes(0) {}
  ~QueryBufferPool() {}

  size_t getTotalUsedBytes() const { return _totalUsedBytes; }

 private:
  std::shared_ptr<T> _initializeRsrc(size_t numBytes) final {
    auto renderer = this->_rendererPtr.lock();
    CHECK(renderer);
    _totalUsedBytes += numBytes;
    return std::make_shared<T>(renderer.get(), numBytes);
  }

  void _updateRsrc(std::shared_ptr<T>& rsrcPtr, size_t numBytes) final {
    auto currBytes = rsrcPtr->getNumBytes();
    rsrcPtr->rebuild(numBytes);
    if (currBytes > numBytes) {
      _totalUsedBytes -= (currBytes - numBytes);
    } else {
      _totalUsedBytes += (numBytes - currBytes);
    }
  }

  void _deleteRsrc(const T* deletedRsrcPtr) final { _totalUsedBytes -= deletedRsrcPtr->getNumBytes(); }

  size_t _totalUsedBytes;
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYBUFFERPOOL_H_
