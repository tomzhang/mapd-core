#ifndef QUERYRENDERER_QUERYIDMAPPBOPOOL_H_
#define QUERYRENDERER_QUERYIDMAPPBOPOOL_H_

#include "Types.h"
#include "QueryIdMapPixelBuffer.h"
#include "BaseQueryRsrcPool.h"

namespace QueryRenderer {

template <typename T = unsigned int>
class QueryIdMapPboPool : public BaseQueryRsrcPool<QueryIdMapPixelBuffer<T>, 300000, size_t, size_t> {
 public:
  QueryIdMapPboPool(::Rendering::GL::GLRendererShPtr& renderer)
      : BaseQueryRsrcPool<QueryIdMapPixelBuffer<T>, 300000, size_t, size_t>(renderer) {}
  ~QueryIdMapPboPool() {}

 private:
  std::shared_ptr<QueryIdMapPixelBuffer<T>> _initializeRsrc(size_t width, size_t height) final {
    auto renderer = this->_rendererPtr.lock();
    CHECK(renderer);
    return std::make_shared<QueryIdMapPixelBuffer<T>>(renderer.get(), width, height);
  }

  void _updateRsrc(std::shared_ptr<QueryIdMapPixelBuffer<T>>& rsrcPtr, size_t width, size_t height) final {
    rsrcPtr->resize(width, height);
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_QUERYIDMAPPBOPOOL_H_
