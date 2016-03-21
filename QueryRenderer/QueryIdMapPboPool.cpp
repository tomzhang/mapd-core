#include "QueryIdMapPboPool.h"
#include "QueryIdMapPixelBuffer.h"
#include <Rendering/RenderError.h>
#include <Rendering/Renderer/GL/GLRenderer.h>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLRendererShPtr;
using ::Rendering::GL::GLRendererWkPtr;

QueryIdMapPboPool::QueryIdMapPboPool(GLRendererShPtr& renderer) : _rendererPtr(renderer) {
}

QueryIdMapPboPool::~QueryIdMapPboPool() {
}

QueryIdMapPixelBufferShPtr QueryIdMapPboPool::getInactiveIdMapPbo(size_t width, size_t height) {
  std::lock_guard<std::mutex> pool_lock(_poolMtx);

  QueryIdMapPixelBufferShPtr rtn;

  if (_inactivePboQueue.size() == 0) {
    GLRendererShPtr renderer = _rendererPtr.lock();
    CHECK(renderer);
    rtn.reset(new QueryIdMapPixelBuffer(renderer.get(), width, height));
    _pboMap.emplace(rtn);
  } else {
    rtn = _inactivePboQueue.front();
    _inactivePboQueue.pop_front();
    rtn->resize(width, height);
  }

  return rtn;
}

void QueryIdMapPboPool::setIdMapPboInactive(QueryIdMapPixelBufferShPtr& pbo) {
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

}  // namespace QueryRenderer
