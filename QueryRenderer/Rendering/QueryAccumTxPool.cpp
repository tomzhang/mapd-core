#include "QueryAccumTxPool.h"
#include "QueryRenderCompositor.h"
#include "shaders/accumulator_passThru_vert.h"
#include "shaders/accumulatorTx_findExtents_frag.h"

#include <Rendering/RenderError.h>
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <Rendering/Objects/Array2d.h>

#include <limits>

#include <Shared/measure.h>
#include <iostream>

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
        auto accumItem = dynamic_cast<const AccumTxContainer*>(item.get());
        CHECK(accumItem);
        _compositorPtr->unregisterAccumulatorTexture(accumItem->tx, accumItem->accumTxIdx);
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
  auto txItr = _accumTxMap.emplace(new AccumTxContainer(txPtr, idx)).first;
  _inactiveAccumTxQueue.insert(dynamic_cast<const AccumTxContainer*>(txItr->get()));

  if (_compositorPtr) {
    _compositorPtr->registerAccumulatorTexture(txPtr, idx, numTextures);
  }
}

void QueryAccumTxPool::_initialize(size_t width, size_t height, size_t numTextures) {
  // NOTE: call of this function must lock the _poolMtx

  GLRendererShPtr renderer = _rendererPtr.lock();
  CHECK(renderer);
  auto rsrcMgr = renderer->getResourceManager();

  if (numTextures > 0) {
    if (!clearPboPtr) {
      ::Rendering::Objects::Array2d<unsigned int> clearData(width, height, 0);
      clearPboPtr =
          rsrcMgr->createPixelBuffer2d(width, height, GL_RED_INTEGER, GL_UNSIGNED_INT, clearData.getDataPtr());
    }
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
  _inactiveAccumTxQueue.insert(dynamic_cast<const AccumTxContainer*>(itr->get()));
}

void QueryAccumTxPool::setAccumTxInactive(const std::vector<GLTexture2dShPtr>& txs) {
  std::lock_guard<std::mutex> pool_lock(_poolMtx);

  _cleanupUnusedTxs();

  for (auto& accumTx : txs) {
    auto itr = _accumTxMap.find(accumTx);
    CHECK(itr != _accumTxMap.end());

    _accumTxMap.modify(itr, ChangeLastUsedTime());
    _inactiveAccumTxQueue.insert(dynamic_cast<const AccumTxContainer*>(itr->get()));
  }
}

void QueryAccumTxPool::_runExtentsPass(GLRendererShPtr& renderer,
                                       GLTexture2dShPtr& extentTx,
                                       ::Rendering::GL::Resources::GLTexture2d* tx,
                                       ::Rendering::GL::Resources::GLTexture2dArray* txArray) {
  // auto clock_begin = timer_start();

  CHECK(clearExtentsPboPtr && extentsShaderPtr);

  extentTx->copyPixelsFromPixelBuffer(clearExtentsPboPtr);

  // TODO(croot): for some reason, I need to unbind the texture2d here
  // otherwise the renderer->getBoundTexture2dPixels() call below
  // gets weird values and crashes. Need to investigate why, but
  // for now, unbinding/rebinding seems to work. One thing to check
  // is if the bound texture2d is set underneath us somehow
  renderer->bindTexture2d(nullptr);

  renderer->bindShader(extentsShaderPtr);

  extentsShaderPtr->setImageLoadStoreAttribute("inExtents", extentTx);

  if (tx) {
    extentsShaderPtr->setSamplerTextureImageUnit("inTxPixelCounter", GL_TEXTURE0);
    extentsShaderPtr->setSamplerAttribute("inTxPixelCounter", tx);
    extentsShaderPtr->setSubroutine("getAccumulatedCnt", "getTxAccumulatedCnt");
  } else if (txArray) {
    extentsShaderPtr->setSamplerTextureImageUnit("inTxArrayPixelCounter", GL_TEXTURE1);
    extentsShaderPtr->setSamplerAttribute("inTxArrayPixelCounter", txArray);
    extentsShaderPtr->setSubroutine("getAccumulatedCnt", "getTxArrayAccumulatedCnt");
  }

  // TODO(croot): binding the vao here should still work as it has the
  // same vbo bindings as the accumulator2ndPassShader, but
  // this subverts the vao creation API, which implies a binding with
  // a specific shader, or does it?
  renderer->bindVertexArray(vao);
  renderer->drawVertexBuffers(GL_TRIANGLE_STRIP);

  // std::vector<unsigned int> extents({0, 0});

  // auto extents_time_ms = timer_stop(clock_begin);

  // renderer->bindTexture2d(rtn);
  // renderer->getBoundTexture2dPixels(2, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &extents[0]);
  // std::cerr << "CROOT - the extents - min: " << extents[0] << ", " << extents[1] << " - time: " << extents_time_ms
  //           << " ms" << std::endl;
}

::Rendering::GL::Resources::GLTexture2dShPtr QueryAccumTxPool::_getInactiveExtentsTxInternal(
    ::Rendering::GL::Resources::GLTexture2d* tx,
    ::Rendering::GL::Resources::GLTexture2dArray* txArray) {
  // NOTE: the pool mutex must be locked before calling this function
  GLRendererShPtr renderer = _rendererPtr.lock();
  CHECK(renderer);
  GLTexture2dShPtr rtn;

  if (!clearExtentsPboPtr) {
    CHECK(!extentsShaderPtr);

    auto rsrcMgr = renderer->getResourceManager();

    static std::vector<uint32_t> initExtents(
        {std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::min()});

    clearExtentsPboPtr = rsrcMgr->createPixelBuffer2d(2, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &initExtents[0]);
    extentsShaderPtr = rsrcMgr->createShader(Accumulator_PassThru_vert::source, AccumulatorTx_FindExtents_frag::source);
  }

  if (!rectvbo) {
    CHECK(!vao && extentsShaderPtr);
    auto rsrcMgr = renderer->getResourceManager();

    ::Rendering::GL::Resources::GLInterleavedBufferLayoutShPtr bufferLayout(
        new ::Rendering::GL::Resources::GLInterleavedBufferLayout());
    bufferLayout->addAttribute<float, 2>("pos");

    rectvbo = rsrcMgr->createVertexBuffer<float>({-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0}, bufferLayout);
    renderer->bindShader(extentsShaderPtr);
    vao = rsrcMgr->createVertexArray({{rectvbo, {}}});
  }

  if (_inactiveExtentTxQueue.size() == 0) {
    auto rsrcMgr = renderer->getResourceManager();
    rtn = rsrcMgr->createTexture2d(2, 1, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 1);
    _extentTxMap.emplace(new TxContainer(rtn));
  } else {
    rtn = _inactiveExtentTxQueue.front();
    _inactiveExtentTxQueue.pop_front();
  }

  _runExtentsPass(renderer, rtn, tx, txArray);

  return rtn;
}

::Rendering::GL::Resources::GLTexture2dShPtr QueryAccumTxPool::getInactiveExtentsTx(
    ::Rendering::GL::Resources::GLTexture2dShPtr& tx) {
  std::lock_guard<std::mutex> pool_lock(_poolMtx);

  CHECK(_accumTxMap.find(tx) != _accumTxMap.end());

  return _getInactiveExtentsTxInternal(tx.get(), nullptr);
}

::Rendering::GL::Resources::GLTexture2dShPtr QueryAccumTxPool::getInactiveExtentsTx(
    ::Rendering::GL::Resources::GLTexture2dArray* txArray) {
  std::lock_guard<std::mutex> pool_lock(_poolMtx);

  return _getInactiveExtentsTxInternal(nullptr, txArray);
}

void QueryAccumTxPool::setExtentsTxInactive(::Rendering::GL::Resources::GLTexture2dShPtr& tx) {
  std::lock_guard<std::mutex> pool_lock(_poolMtx);

  auto itr = _extentTxMap.find(tx);
  CHECK(itr != _extentTxMap.end());

  _extentTxMap.modify(itr, ChangeLastUsedTime());

  // delete pbos that have been inactive for a while
  static const std::chrono::milliseconds maxIdleTime = std::chrono::milliseconds(300000);  // 5 minutes
  std::chrono::milliseconds cutoffTime = getCurrentTimeMS() - maxIdleTime;

  while (_inactiveExtentTxQueue.begin() != _inactiveExtentTxQueue.end()) {
    itr = _extentTxMap.find(_inactiveExtentTxQueue.front());
    CHECK(itr != _extentTxMap.end());

    if ((*itr)->lastUsedTime >= cutoffTime) {
      break;
    }

    _extentTxMap.erase(itr);
    _inactiveExtentTxQueue.pop_front();
  }

  _inactiveExtentTxQueue.push_back(tx);
}

void QueryAccumTxPool::resize(size_t width, size_t height) {
  size_t myWidth, myHeight;

  for (auto& item : _accumTxMap) {
    myWidth = std::max(width, item->tx->getWidth());
    myHeight = std::max(height, item->tx->getHeight());

    item->tx->resize(myWidth, myHeight);
  }
}

}  // namespace QueryRenderer
