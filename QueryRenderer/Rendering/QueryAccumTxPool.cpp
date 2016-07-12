#include "QueryAccumTxPool.h"
#include "QueryRenderCompositor.h"
#include <Rendering/RenderError.h>
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Objects/Array2d.h>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLRendererShPtr;
using ::Rendering::GL::GLRendererWkPtr;
using ::Rendering::GL::Resources::GLTexture2dShPtr;

bool QueryAccumTxPool::containerCmp(const AccumTxContainer* a, const AccumTxContainer* b) {
  return a->accumTxIdx < b->accumTxIdx;
}

const size_t QueryAccumTxPool::maxTextures = 500;

QueryAccumTxPool::QueryAccumTxPool(GLRendererShPtr& renderer, QueryRenderCompositorShPtr compositorPtr)
    : _rendererPtr(renderer), _compositorPtr(compositorPtr), _inactiveAccumTxQueue(&QueryAccumTxPool::containerCmp) {
  CHECK(renderer);
}

QueryAccumTxPool::~QueryAccumTxPool() {
  if (_accumTxMap.size() || clearPboPtr) {
    if (_compositorPtr) {
      for (auto& item : _accumTxMap) {
        _compositorPtr->unregisterAccumulatorTexture(item.tx, item.accumTxIdx);
      }
    }
    GLRendererShPtr renderer = _rendererPtr.lock();
    if (renderer) {
      // make the renderer active to delete all the resources
      renderer->makeActiveOnCurrentThread();
    }
  }
}

void QueryAccumTxPool::_createTx(::Rendering::GL::GLResourceManagerShPtr& rsrcMgr,
                                 size_t width,
                                 size_t height,
                                 size_t idx,
                                 size_t numTextures) {
  RUNTIME_EX_ASSERT(_accumTxMap.size() < maxTextures, "Max number of accumulation textures in pool reached!");
  auto txPtr = rsrcMgr->createTexture2d(width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1);
  auto txItr = _accumTxMap.emplace(txPtr, idx).first;
  _inactiveAccumTxQueue.insert(&(*txItr));

  if (_compositorPtr) {
    _compositorPtr->registerAccumulatorTexture(txPtr, idx, numTextures);
  }
}

void QueryAccumTxPool::_initialize(size_t width, size_t height, size_t numTextures) {
  // NOTE: call of this function must lock the _poolMtx

  GLRendererShPtr renderer = _rendererPtr.lock();
  CHECK(renderer);
  auto rsrcMgr = renderer->getResourceManager();

  if (numTextures > 0 && !clearPboPtr) {
    ::Rendering::Objects::Array2d<unsigned int> clearData(width, height, 0);
    clearPboPtr = rsrcMgr->createPixelBuffer2d(width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, clearData.getDataPtr());
  }

  auto itr = _inactiveAccumTxQueue.begin();
  for (size_t i = 0; i < numTextures; ++i) {
    while (itr != _inactiveAccumTxQueue.end() && (*itr)->accumTxIdx < i) {
      itr++;
    }

    // TODO(croot): register the new textures with the compositor
    if (itr == _inactiveAccumTxQueue.end()) {
      for (size_t j = i; j < numTextures; ++j) {
        _createTx(rsrcMgr, width, height, j, numTextures);
      }
      break;
    } else if ((*itr)->accumTxIdx > i) {
      _createTx(rsrcMgr, width, height, i, numTextures);
    } else {
      itr++;
    }
  }

  // int diff = static_cast<int>(numTextures) - static_cast<int>(_inactiveAccumTxQueue.size());
  // if (diff > 0) {
  //   GLRendererShPtr renderer = _rendererPtr.lock();
  //   CHECK(renderer);
  //   auto rsrcMgr = renderer->getResourceManager();

  //   if (!clearPboPtr) {
  //     ::Rendering::Objects::Array2d<unsigned int> clearData(width, height, 0);
  //     clearPboPtr =
  //         rsrcMgr->createPixelBuffer2d(width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, clearData.getDataPtr());
  //   }

  //   // NOTE: the accumulation buffer must have only 1 sample due to the
  //   // EGLImage restriction
  //   GLTexture2dShPtr txPtr;
  //   for (int i = 0; i < diff; ++i) {
  //     // NOTE: the accumulation buffer must have only 1 sample due to the
  //     // EGLImage restriction
  //     txPtr = rsrcMgr->createTexture2d(width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1);
  //     _accumTxMap.emplace(txPtr);
  //     _inactiveAccumTxQueue.push_back(txPtr);
  //   }
  // }
}

void QueryAccumTxPool::_cleanupUnusedTxs() {
  // NOTE: caller of this functino must lock _poolMtx

  // delete accumTxs that have been inactive for a while
  static const std::chrono::milliseconds maxAccumTxIdleTime = std::chrono::milliseconds(300000);  // 5 minutes
  std::chrono::milliseconds cutoffTime = getCurrentTimeMS() - maxAccumTxIdleTime;

  auto itr = _inactiveAccumTxQueue.begin();
  decltype(itr) nextitr;
  while (itr != _inactiveAccumTxQueue.end()) {
    if ((*itr)->lastUsedTime < cutoffTime) {
      auto accumItr = _accumTxMap.find((*itr)->tx);
      if (_compositorPtr) {
        _compositorPtr->unregisterAccumulatorTexture((*itr)->tx, (*itr)->accumTxIdx);
      }
      CHECK(accumItr != _accumTxMap.end());
      _accumTxMap.erase(accumItr);

      nextitr = itr;
      nextitr++;
      _inactiveAccumTxQueue.erase(itr);
      itr = nextitr;
    } else {
      itr++;
    }
  }
}

GLTexture2dShPtr QueryAccumTxPool::getInactiveAccumTx(size_t width, size_t height) {
  std::lock_guard<std::mutex> pool_lock(_poolMtx);

  GLTexture2dShPtr rtn;

  _initialize(width, height, 1);

  auto itr = _inactiveAccumTxQueue.begin();
  CHECK(itr != _inactiveAccumTxQueue.end() && (*itr)->accumTxIdx == 0);

  rtn = (*itr)->tx;
  _inactiveAccumTxQueue.erase(itr);

  auto widthToUse = std::max(width, rtn->getWidth());
  auto heightToUse = std::max(height, rtn->getHeight());

  if (widthToUse > clearPboPtr->getWidth() || heightToUse > clearPboPtr->getHeight()) {
    ::Rendering::Objects::Array2d<unsigned int> clearData(widthToUse, heightToUse, 0);
    clearPboPtr->resize(widthToUse, heightToUse, clearData.getDataPtr());
    rtn->resize(widthToUse, heightToUse);
  }

  // TODO(croot): clear the texture with the clear pbo before returning
  rtn->copyPixelsFromPixelBuffer(clearPboPtr, 0, 0, width, height);

  return rtn;
}

std::vector<GLTexture2dShPtr> QueryAccumTxPool::getInactiveAccumTx(size_t width, size_t height, size_t numTextures) {
  std::lock_guard<std::mutex> pool_lock(_poolMtx);

  std::vector<GLTexture2dShPtr> rtn(numTextures);

  _initialize(width, height, numTextures);

  CHECK(clearPboPtr);

  auto widthToUse = std::max(width, clearPboPtr->getWidth());
  auto heightToUse = std::max(height, clearPboPtr->getHeight());

  if (widthToUse > clearPboPtr->getWidth() || heightToUse > clearPboPtr->getHeight()) {
    ::Rendering::Objects::Array2d<unsigned int> clearData(widthToUse, heightToUse, 0);
    clearPboPtr->resize(widthToUse, heightToUse, clearData.getDataPtr());
  }

  GLTexture2dShPtr txPtr;
  decltype(_inactiveAccumTxQueue)::iterator itr, nextitr;
  itr = _inactiveAccumTxQueue.begin();

  for (size_t i = 0; i < numTextures; ++i) {
    while (itr != _inactiveAccumTxQueue.end() && (*itr)->accumTxIdx < i) {
      itr++;
    }
    CHECK(itr != _inactiveAccumTxQueue.end() && (*itr)->accumTxIdx == i);

    txPtr = (*itr)->tx;

    nextitr = itr;
    nextitr++;
    _inactiveAccumTxQueue.erase(itr);
    itr = nextitr;

    widthToUse = std::max(width, txPtr->getWidth());
    heightToUse = std::max(height, txPtr->getHeight());

    txPtr->resize(widthToUse, heightToUse);

    // TODO(croot): clear the texture with the clear pbo before returning
    txPtr->copyPixelsFromPixelBuffer(clearPboPtr, 0, 0, width, height);

    rtn[i] = txPtr;
  }

  return rtn;
}

void QueryAccumTxPool::setAccumTxInactive(GLTexture2dShPtr& accumTx) {
  std::lock_guard<std::mutex> pool_lock(_poolMtx);

  _cleanupUnusedTxs();

  auto itr = _accumTxMap.find(accumTx);
  CHECK(itr != _accumTxMap.end());

  _accumTxMap.modify(itr, ChangeLastUsedTime());
  _inactiveAccumTxQueue.insert(&(*itr));
}

void QueryAccumTxPool::setAccumTxInactive(const std::vector<GLTexture2dShPtr>& txs) {
  std::lock_guard<std::mutex> pool_lock(_poolMtx);

  _cleanupUnusedTxs();

  for (auto& accumTx : txs) {
    auto itr = _accumTxMap.find(accumTx);
    CHECK(itr != _accumTxMap.end());

    _accumTxMap.modify(itr, ChangeLastUsedTime());
    _inactiveAccumTxQueue.insert(&(*itr));
  }
}

void QueryAccumTxPool::resize(size_t width, size_t height) {
  size_t myWidth, myHeight;

  for (auto& item : _accumTxMap) {
    myWidth = std::max(width, item.tx->getWidth());
    myHeight = std::max(height, item.tx->getHeight());

    item.tx->resize(myWidth, myHeight);
  }
}

}  // namespace QueryRenderer
