#include "QueryRenderer.h"
#include "QueryFramebuffer.h"
#include "QueryDataTable.h"
#include "QueryRendererObjects.h"
#include "QueryRenderCompositor.h"
#include <Rendering/Window.h>
#include <Rendering/Renderer/GL/GLRenderer.h>

#include <png.h>
#include <vector>
#include <thread>
// #include "MapDGL.h"
// #include "QueryRenderer.h"
// #include "RapidJSONUtils.h"
// // #include "../QueryEngine/Execute.h"
// #include <glog/logging.h>
// #include <utility>  // std::pair
// #include <unordered_set>
// #include "rapidjson/document.h"
// #include "rapidjson/pointer.h"
#include "rapidjson/error/en.h"
// #include "rapidjson/allocators.h"

namespace QueryRenderer {

using ::Rendering::Renderer;
using ::Rendering::WindowShPtr;
using ::Rendering::RendererShPtr;
using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::Resources::FboBind;

QueryRenderer::QueryRenderer(bool doHitTest, bool doDepthTest)
    : _ctx(new QueryRendererContext(doHitTest, doDepthTest)), _perGpuData(), _compositorPtr(nullptr) {
}

QueryRenderer::QueryRenderer(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                             bool doHitTest,
                             bool doDepthTest)
    : QueryRenderer(doHitTest, doDepthTest) {
  _initFromJSON(jsonDocumentPtr);
}

QueryRenderer::QueryRenderer(const std::string& configJSON, bool doHitTest, bool doDepthTest)
    : QueryRenderer(doHitTest, doDepthTest) {
  _initFromJSON(configJSON);
}

QueryRenderer::~QueryRenderer() {
  _clear();
  _clearGpuResources();
}

void QueryRenderer::_clear() {
  _ctx->_clear();
}

void QueryRenderer::_clearGpuResources() {
  _perGpuData.clear();
}

void QueryRenderer::_updateGpuData(const GpuId& gpuId,
                                   QueryRenderManager::PerGpuDataMap& qrmPerGpuData,
                                   std::unordered_set<GpuId>& unusedGpus,
                                   bool update) {
  auto myItr = _perGpuData.find(gpuId);
  if (myItr == _perGpuData.end()) {
    auto itr = qrmPerGpuData.find(gpuId);
    CHECK(itr != qrmPerGpuData.end());

    PerGpuData gpuData;

    gpuData.qrmGpuData = itr->second;

    // TODO(croot): validate the QueryRenderManager data is complete?
    CHECK(itr->second->rendererPtr != nullptr);

    if (update) {
      GLRenderer* renderer = dynamic_cast<GLRenderer*>(itr->second->rendererPtr.get());
      CHECK(renderer != nullptr);

      itr->second->makeActiveOnCurrentThread();

      if (_compositorPtr) {
        gpuData.framebufferPtr.reset(new QueryFramebuffer(_compositorPtr.get(), renderer));
      } else {
        gpuData.framebufferPtr.reset(new QueryFramebuffer(
            renderer, _ctx->getWidth(), _ctx->getHeight(), _ctx->doHitTest(), _ctx->doDepthTest()));
      }
    }

    _perGpuData.emplace(gpuId, std::move(gpuData));
  } else {
    myItr->second.makeActiveOnCurrentThread();
    myItr->second.framebufferPtr->resize(_ctx->getWidth(), _ctx->getHeight());

    unusedGpus.erase(gpuId);
  }
}

std::unordered_set<GpuId> QueryRenderer::_initUnusedGpus() {
  std::unordered_set<GpuId> unusedGpus;
  unusedGpus.reserve(_perGpuData.size());
  for (const auto& kv : _perGpuData) {
    unusedGpus.insert(kv.first);
  }
  return unusedGpus;
}

void QueryRenderer::_initGpuResources(QueryRenderManager::PerGpuDataMap& qrmPerGpuData,
                                      const std::vector<GpuId>& gpuIds,
                                      std::unordered_set<GpuId>& unusedGpus) {
  bool update = (_perGpuData.size() > 0);
  if (gpuIds.size()) {
    for (auto gpuId : gpuIds) {
      _updateGpuData(gpuId, qrmPerGpuData, unusedGpus, update);
    }
  } else {
    for (auto& qrmItr : qrmPerGpuData) {
      _updateGpuData(qrmItr.first, qrmPerGpuData, unusedGpus, update);
    }
  }

  // now clean up any unused gpu resources
  for (auto gpuId : unusedGpus) {
    _perGpuData.erase(gpuId);
  }

  _ctx->_initGpuResources(_perGpuData, unusedGpus);

  if (_compositorPtr) {
    _compositorPtr->cleanupUnusedFbos();
  }
}

void QueryRenderer::_resizeFramebuffers(int width, int height) {
  // TODO(croot): the compositor will need to be pulled out to the QueryRenderManager
  // level if/when we store a large framebuffer for all connected users to use.
  QueryRenderManager::PerGpuDataShPtr qrmGpuData;

  if (_perGpuData.size() > 1) {
    // NOTE: since we're using a map to store the per gpu data, the first item
    // will be the lowest active gpu id.
    auto itr = _perGpuData.begin();
    itr->second.makeActiveOnCurrentThread();
    if (!_compositorPtr) {
      qrmGpuData = itr->second.getQRMGpuData();
      _compositorPtr.reset(new QueryRenderCompositor(this,
                                                     qrmGpuData->rendererPtr,
                                                     width,
                                                     height,
                                                     // TODO(croot): should we support num samples
                                                     1,  //_ctx->getNumSamples(),
                                                     _ctx->doHitTest(),
                                                     _ctx->doDepthTest()));
    } else {
      // TODO(croot): we need to verify that the compositor is on a GPU that is
      // still active for this renderer. If not, then we need to build a new compositor
      // and update all existing framebuffers to refer to this new compositor.
      // NOTE: we won't have to do this TODO if we move the framebuffers and compositors
      // to the QueryRenderManager level.
      // if (_compositorPtr->getRenderer() != itr->second.qrmGpuData->rendererPtr.get()) {
      // }
      _compositorPtr->resize(width, height);
    }
  } else {
    _compositorPtr.reset(nullptr);
  }

  for (auto& gpuDataItr : _perGpuData) {
    gpuDataItr.second.makeActiveOnCurrentThread();

    if (gpuDataItr.second.framebufferPtr == nullptr) {
      qrmGpuData = gpuDataItr.second.getQRMGpuData();
      CHECK(qrmGpuData->rendererPtr != nullptr);
      GLRenderer* renderer = dynamic_cast<GLRenderer*>(qrmGpuData->rendererPtr.get());
      CHECK(renderer != nullptr);

      if (_compositorPtr) {
        gpuDataItr.second.framebufferPtr.reset(new QueryFramebuffer(_compositorPtr.get(), renderer));
      } else {
        gpuDataItr.second.framebufferPtr.reset(
            new QueryFramebuffer(renderer, width, height, _ctx->doHitTest(), _ctx->doDepthTest()));
      }
    } else {
      gpuDataItr.second.framebufferPtr->resize(width, height);
    }
  }
}

void QueryRenderer::_initFromJSON(const std::string& configJSON, bool forceUpdate) {
  std::shared_ptr<rapidjson::Document> objPtr(new rapidjson::Document());

  objPtr->Parse(configJSON.c_str());

  // TODO(croot): this can be removed if the executor will handle the initial parse.
  RUNTIME_EX_ASSERT(!objPtr->HasParseError(),
                    "JSON parse error - " + std::to_string(objPtr->GetErrorOffset()) + ", error: " +
                        rapidjson::GetParseError_En(objPtr->GetParseError()));

  _initFromJSON(objPtr);
}

void QueryRenderer::_initFromJSON(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate) {
  rapidjson::Pointer rootPath;
  rapidjson::Value* obj = jsonDocumentPtr.get();

  if (forceUpdate) {
    _clear();
    if (!_ctx->_jsonCache->ObjectEmpty()) {
      _ctx->_jsonCache.reset(new rapidjson::Value());
    }
  }

  if (_ctx->_jsonCache && *_ctx->_jsonCache == *jsonDocumentPtr) {
    // nothing's changed. Return.
    // std::cerr << "CROOT - cache hasn't changed!" << std::endl;
    return;
  }

  RUNTIME_EX_ASSERT(obj->IsObject(), "JSON parse error - Root object is not a JSON object.");

  rapidjson::Value::ConstMemberIterator mitr;
  rapidjson::Value::ConstValueIterator vitr;

  RUNTIME_EX_ASSERT((mitr = obj->FindMember("width")) != obj->MemberEnd(),
                    "JSON parse error - \"width\" is not defined.");
  RUNTIME_EX_ASSERT(mitr->value.IsInt(), "JSON parse error - \"width\" is not an integer.");
  int width = mitr->value.GetInt();

  RUNTIME_EX_ASSERT((mitr = obj->FindMember("height")) != obj->MemberEnd(),
                    "JSON parse error - \"height\" is not defined.");
  RUNTIME_EX_ASSERT(mitr->value.IsInt(), "JSON parse error - \"height\" is not an integer.");
  int height = mitr->value.GetInt();

  setWidthHeight(width, height);

  std::string propName = "data";
  mitr = obj->FindMember(propName.c_str());
  if (mitr != obj->MemberEnd()) {
    rapidjson::Pointer dataPath = rootPath.Append(propName.c_str(), propName.length());

    RUNTIME_EX_ASSERT(mitr->value.IsArray(), "JSON parse error - the \"" + propName + "\" member must be an array.");

    QueryDataTableVBOShPtr dataTablePtr;
    std::unordered_set<std::string> visitedNames;
    std::unordered_set<std::string> unvisitedNames;
    unvisitedNames.reserve(_ctx->_dataTableMap.size());
    for (auto kv : _ctx->_dataTableMap) {
      unvisitedNames.insert(kv.first);
    }

    for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
      rapidjson::Pointer dataObjPath = dataPath.Append(vitr - mitr->value.Begin());

      std::string tableName = getDataTableNameFromJSONObj(*vitr);

      RUNTIME_EX_ASSERT(visitedNames.find(tableName) == visitedNames.end(),
                        "JSON parse error - a data table with the name \"" + tableName + "\" already exists.");

      dataTablePtr = _ctx->getDataTable(tableName);

      if (!dataTablePtr) {
        dataTablePtr = createDataTable(*vitr, dataObjPath, _ctx, tableName);
        _ctx->_dataTableMap.insert(std::make_pair(tableName, dataTablePtr));
      } else {
        // TODO(croot): data table is changing. Need to validate any previously existing references.
        // One way to do this is store a map of all objects changing in-place in order to
        // validate.
        if (dataTablePtr->getType() != getDataTableTypeFromJSONObj(*vitr)) {
          // completely new data table type, so destroy previous one and
          // build a new one from scratch.
          _ctx->_dataTableMap.erase(tableName);
          dataTablePtr = createDataTable(*vitr, dataObjPath, _ctx, tableName);
          _ctx->_dataTableMap.insert(std::make_pair(tableName, dataTablePtr));
        } else {
          dataTablePtr->updateFromJSONObj(*vitr, dataObjPath);
        }
      }

      unvisitedNames.erase(tableName);
      visitedNames.insert(std::move(tableName));
    }

    // now remove any unused tables that may be lingering around
    for (const auto& itr : unvisitedNames) {
      _ctx->_dataTableMap.erase(itr);
    }
  } else {
    // need to clear out the previous data
    // TODO(croot): Need to invalidate any previous data references
    // This should probably be handled by some data reference object.
    // That or do an object validation check after everything's been rebuilt.
    // The latter would be the easiest way.
    _ctx->_dataTableMap.clear();
  }

  std::array<std::vector<ScaleShPtr>, static_cast<size_t>(RefEventType::ALL)> scaleEvents;
  propName = "scales";
  mitr = obj->FindMember(propName.c_str());
  if (mitr != obj->MemberEnd()) {
    rapidjson::Pointer scalePath = rootPath.Append(propName.c_str(), propName.length());

    RUNTIME_EX_ASSERT(mitr->value.IsArray(), "JSON parse error - the \"" + propName + "\" member must be an array.");

    ScaleShPtr scalePtr;
    std::unordered_set<std::string> visitedNames;
    std::unordered_set<std::string> unvisitedNames;
    unvisitedNames.reserve(_ctx->_scaleConfigMap.size());
    for (auto kv : _ctx->_scaleConfigMap) {
      unvisitedNames.insert(kv.first);
    }

    for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
      rapidjson::Pointer scaleObjPath = scalePath.Append(vitr - mitr->value.Begin());

      std::string scaleName = getScaleNameFromJSONObj(*vitr);

      RUNTIME_EX_ASSERT(visitedNames.find(scaleName) == visitedNames.end(),
                        "JSON parse error - a scale with the name \"" + scaleName + "\" already exists.");

      scalePtr = _ctx->getScale(scaleName);

      if (!scalePtr) {
        scalePtr = createScale(*vitr, scaleObjPath, _ctx, scaleName);
        _ctx->_scaleConfigMap.insert(std::make_pair(scaleName, scalePtr));

        // TODO(croot): add an Add event type?
      } else {
        // TODO(croot): scale config is changing. Need to validate any previously existing references.
        // One way to do this is store a map of all objects changing in-place in order to
        // validate.
        if (scalePtr->getType() != getScaleTypeFromJSONObj(*vitr) ||
            scalePtr->getDomainDataType() != getScaleDomainDataTypeFromJSONObj(*vitr, _ctx) ||
            scalePtr->getRangeDataType() != getScaleRangeDataTypeFromJSONObj(*vitr, _ctx)) {
          // completely new scale type, so destroy previous one and
          // build a new one from scratch.
          _ctx->_scaleConfigMap.erase(scaleName);
          scalePtr = createScale(*vitr, scaleObjPath, _ctx, scaleName);
          _ctx->_scaleConfigMap.insert(std::make_pair(scaleName, scalePtr));

          scaleEvents[static_cast<size_t>(RefEventType::REPLACE)].push_back(scalePtr);
        } else {
          if (scalePtr->updateFromJSONObj(*vitr, scaleObjPath)) {
            scaleEvents[static_cast<size_t>(RefEventType::UPDATE)].push_back(scalePtr);
          }
        }
      }

      unvisitedNames.erase(scaleName);
      visitedNames.insert(std::move(scaleName));
    }

    // now remove any unused scales that may be lingering around
    for (const auto& unvisitedName : unvisitedNames) {
      scalePtr = _ctx->getScale(unvisitedName);
      _ctx->_scaleConfigMap.erase(unvisitedName);
      scaleEvents[static_cast<size_t>(RefEventType::REMOVE)].push_back(scalePtr);
    }

  } else {
    // need to clear out the previous data
    // TODO(croot): Need to invalidate any previous data references
    // This should probably be handled by some scale reference object.
    // That or do an object validation check after everything's been rebuilt.
    // The latter would be the easiest way.
    _ctx->_scaleConfigMap.clear();
  }

  propName = "marks";
  mitr = obj->FindMember(propName.c_str());
  if (mitr != obj->MemberEnd()) {
    rapidjson::Pointer markPath = rootPath.Append(propName.c_str(), propName.length());

    RUNTIME_EX_ASSERT(mitr->value.IsArray(), "JSON parse error - the \"" + propName + "\" member must be an array");

    size_t i;
    for (vitr = mitr->value.Begin(), i = 0; vitr != mitr->value.End(); ++vitr, ++i) {
      rapidjson::Pointer markObjPath = markPath.Append(vitr - mitr->value.Begin());

      if (i == _ctx->_geomConfigs.size()) {
        GeomConfigShPtr geomConfigPtr = createMark(*vitr, markObjPath, _ctx);

        _ctx->_geomConfigs.push_back(geomConfigPtr);
      } else {
        // do an update
        if (_ctx->_geomConfigs[i]->getType() != getMarkTypeFromJSONObj(*vitr)) {
          // TODO(croot): need to do a replace
          THROW_RUNTIME_EX("The type of mark " + RapidJSONUtils::getPointerPath(markObjPath) +
                           " has changed in-place in the json. This has yet to be implemented.");
        } else {
          _ctx->_geomConfigs[i]->updateFromJSONObj(*vitr, markObjPath);
        }
      }
    }
  } else {
    _ctx->_geomConfigs.clear();
  }

  // now fire events so dependencies are cleaned and validated
  for (size_t i = 0; i < scaleEvents.size(); ++i) {
    for (auto scalePtr : scaleEvents[i]) {
      _ctx->_fireRefEvent(static_cast<RefEventType>(i), scalePtr);
    }
  }

  _ctx->_jsonCache = jsonDocumentPtr;
}

int QueryRenderer::getWidth() {
  return _ctx->_width;
}

int QueryRenderer::getHeight() {
  return _ctx->_height;
}

void QueryRenderer::setWidthHeight(int width, int height) {
  _ctx->_width = width;
  _ctx->_height = height;

  _resizeFramebuffers(_ctx->_width, _ctx->_height);

  // if (win) {
  //   // pass a window in debug mode

  //   // resize the window
  //   glfwSetWindowSize(win, width, height);

  //   // now get the actual framebuffer dimensions
  //   int CROOTw, CROOTh;
  //   glfwGetWindowSize(win, &CROOTw, &CROOTh);

  //   int w, h;
  //   glfwGetFramebufferSize(win, &w, &h);
  //   std::cerr << "CROOT - setting window width/height: " << width << "x" << height << ", " << w << "x" << h << ", "
  //             << CROOTw << "x" << CROOTh << std::endl;
  //   _ctx->_width = w;
  //   _ctx->_height = h;
  // }
}

const QueryFramebufferUqPtr& QueryRenderer::getFramebuffer(const GpuId& gpuId) {
  PerGpuDataMap::const_iterator itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr != _perGpuData.end(),
                    "Error getting QueryRenderer framebuffer for gpuId " + std::to_string(gpuId) + ".");

  return itr->second.framebufferPtr;
}

void QueryRenderer::setJSONConfig(const std::string& configJSON, bool forceUpdate) {
  _initFromJSON(configJSON, forceUpdate);
}

void QueryRenderer::setJSONDocument(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate) {
  _initFromJSON(jsonDocumentPtr, forceUpdate);
}

#ifdef HAVE_CUDA
void QueryRenderer::updateResultsPostQuery(QueryDataLayoutShPtr& dataLayoutPtr,
                                           const Executor* executor,
                                           QueryRenderManager::PerGpuDataMap& qrmPerGpuData) {
  std::vector<GpuId> gpuIds;
  for (const auto& kv : qrmPerGpuData) {
    if (kv.second->queryResultBufferPtr->getNumUsedBytes() > 0) {
      gpuIds.push_back(kv.first);
    }
  }

  std::unordered_set<GpuId> unusedGpus = _initUnusedGpus();
  _initGpuResources(qrmPerGpuData, gpuIds, unusedGpus);

  // now update the query result buffers
  _ctx->_queryResultBufferLayout = dataLayoutPtr->convertToBufferLayout();
  _ctx->_invalidKey = dataLayoutPtr->invalidKey;
  _ctx->_queryDataLayoutPtr = dataLayoutPtr;
  _ctx->executor_ = executor;

  for (auto gpuId : gpuIds) {
    auto itr = _perGpuData.find(gpuId);
    CHECK(itr != _perGpuData.end());
    itr->second.getQRMGpuData()->queryResultBufferPtr->setBufferLayout(_ctx->_queryResultBufferLayout);
  }

  // now update the gpu resources for data, scale, and geom configs
  // This needs to be done after the queryResultBuffers are updated.
  _ctx->_updateConfigGpuResources(unusedGpus);

  // int numRows = dataLayoutPtr->numRows;
  // _ctx->_queryResultBufferLayout = dataLayoutPtr->convertToBufferLayout();
  // _ctx->_queryResultVBOPtr->updatePostQuery(_ctx->_queryResultBufferLayout, numRows);
  // _ctx->_invalidKey = dataLayoutPtr->invalidKey;
  // _ctx->_queryDataLayoutPtr.reset(dataLayoutPtr);

  // // TODO(croot): Should each mark have its own invalid key? Probably, if we support
  // // multiple sqls.
  // // But if we deal with multiple sqls, then the context probably shouldn't
  // // hold onto the invalidKey - the QueryResultVertexBuffer probably should, but
  // // then how do we deal with setting that uniform?
  // for (size_t i = 0; i < _ctx->_geomConfigs.size(); ++i) {
  //   _ctx->_geomConfigs[i]->setInvalidKey(_ctx->_invalidKey);
  // }
}
#endif  // HAVE_CUDA

void QueryRenderer::activateGpus(QueryRenderManager::PerGpuDataMap& qrmPerGpuData,
                                 const std::vector<GpuId>& gpusToActivate) {
  std::unordered_set<GpuId> unusedGpus = _initUnusedGpus();
  _initGpuResources(qrmPerGpuData, gpusToActivate, unusedGpus);
}

void QueryRenderer::_update() {
  _ctx->_update();
}

void QueryRenderer::renderGpu(GpuId gpuId,
                              PerGpuDataMap* gpuDataMap,
                              QueryRendererContext* ctx,
                              int r,
                              int g,
                              int b,
                              int a) {
  // TODO(croot): make thread safe?

  auto itr = gpuDataMap->find(gpuId);

  CHECK(itr != gpuDataMap->end());

  QueryRenderManager::PerGpuDataShPtr qrmGpuData = itr->second.getQRMGpuData();

  itr->second.makeActiveOnCurrentThread();

  GLRenderer* renderer = dynamic_cast<GLRenderer*>(qrmGpuData->rendererPtr.get());
  CHECK(renderer != nullptr);

  QueryFramebufferUqPtr& framebufferPtr = itr->second.framebufferPtr;

  RUNTIME_EX_ASSERT(
      framebufferPtr != nullptr,
      "QueryRenderer: The framebuffer is not initialized for gpu " + std::to_string(gpuId) + ". Cannot render.");

  framebufferPtr->bindToRenderer(renderer);

  // TODO(croot): enable a push/pop state stack for the renderer state...
  renderer->enable(GL_BLEND);
  renderer->setBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
  // MAPD_CHECK_GL_ERROR(glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO));
  // MAPD_CHECK_GL_ERROR(glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE));
  renderer->setBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

  renderer->setViewport(0, 0, ctx->_width, ctx->_height);
  if (r >= 0 && g >= 0 && b >= 0 && a >= 0) {
    // renderer->setClearColor(0, 0, 0, 0);
    renderer->setClearColor(r, b, g, a);
    renderer->clearAll();
  }

  for (size_t i = 0; i < ctx->_geomConfigs.size(); ++i) {
    ctx->_geomConfigs[i]->draw(renderer, gpuId);
  }

  // NOTE: We're not swapping buffers because we're using pbuffers
  // qrmGpuData->windowPtr->swapBuffers();

  // CROOT testing code
  // int width = ctx->getWidth();
  // int height = ctx->getHeight();
  // std::shared_ptr<unsigned char> pixelsPtr;
  // pixelsPtr = framebufferPtr->readColorBuffer(0, 0, width, height);

  // PngData pngData(width, height, pixelsPtr);
  // pngData.writeToFile("render_" + std::to_string(gpuId) + ".png");

  // renderer->makeInactive();
}

void QueryRenderer::render(bool inactivateRendererOnThread) {
  // update everything marked dirty before rendering
  _update();

  int numGpusToRender = _perGpuData.size();
  if (numGpusToRender) {
    if (numGpusToRender == 1) {
      auto itr = _perGpuData.begin();
      renderGpu(itr->first, &_perGpuData, _ctx.get(), 0, 0, 0, 0);
    } else {
      // TODO(croot) - launch some threads to handle multi-gpu rendering?
      // Then do a composite, or should I do a composite in the renerToPng?

      CHECK(_compositorPtr);

      _compositorPtr->render();

      // auto itr = _perGpuData.begin();
      // int cnt = 0;
      // for (; itr != _perGpuData.end(); ++itr, ++cnt) {
      //   if (cnt == 0) {
      //     renderGpu(itr->first, &_perGpuData, _ctx.get(), 0, 0, 0, 0);
      //   } else {
      //     renderGpu(itr->first, &_perGpuData, _ctx.get(), 0, 0, 0, 1);
      //   }
      // }

      // NOTE: threaded rendering like what is setup in the commented code
      // below is surprisingly slow on greendragon.
      // I'm not sure what the reason is - perhaps the driver has issues
      // with multi-threading like this?
      // I would suspect threaded rendering like this if the contexts were
      // on the same GPU, but not if each rendering context is on a separate
      // GPU... Hmmm.
      //
      // Anyway, doing it sequentially should be fine as OpenGL calls are
      // asynchronous, and we have to do a composite one gpu/layer at a time in
      // the end anyway.
      // std::vector<std::thread> t(numGpusToRender);

      // auto itr = _perGpuData.begin();
      // int idx = 0;
      // for (; itr != _perGpuData.end(); ++itr, ++idx) {
      //   // need to clear out the context on the current thread
      //   // TODO(croot): do this with a simple function call before going thru
      //   // the loop?
      //   itr->second.qrmGpuData->rendererPtr->makeInactive();
      //   if (idx == 0) {
      //     // t[idx] = std::thread([=] { _renderGpu(itr, 0, 0, 0, 0); });
      //     t[idx] = std::thread(renderGpu, itr->first, &_perGpuData, _ctx.get(), 0, 0, 0, 0);
      //   } else {
      //     t[idx] = std::thread(renderGpu, itr->first, &_perGpuData, _ctx.get(), 0, 0, 0, 1);
      //     // t[idx] = std::thread([=] { _renderGpu(itr, 0, 0, 0, 1); });
      //   }
      // }

      // for (idx = 0; idx < numGpusToRender; ++idx) {
      //   t[idx].join();
      // }
    }
  }

  if (inactivateRendererOnThread) {
    GLRenderer* currRenderer = GLRenderer::getCurrentThreadRenderer();
    currRenderer->makeInactive();
  }
}

PngData QueryRenderer::renderToPng(int compressionLevel) {
  RUNTIME_EX_ASSERT(compressionLevel >= -1 && compressionLevel <= 9,
                    "Invalid compression level " + std::to_string(compressionLevel) + ". It must be a " +
                        "value between 0 (no zlib compression) to 9 (most zlib compression), or -1 (use default).");

  render(false);

  int width = getWidth();
  int height = getHeight();
  std::shared_ptr<unsigned char> pixelsPtr;

  if (_perGpuData.size() > 1) {
    Renderer* renderer = _compositorPtr->getRenderer();
    renderer->makeActiveOnCurrentThread();
    pixelsPtr = _compositorPtr->readColorBuffer(0, 0, width, height);
    renderer->makeInactive();
  } else {
    auto itr = _perGpuData.begin();

    QueryRenderManager::PerGpuDataShPtr qrmGpuData = itr->second.getQRMGpuData();

    // const GpuId& gpuId = itr->first;
    itr->second.makeActiveOnCurrentThread();
    GLRenderer* renderer = dynamic_cast<GLRenderer*>(qrmGpuData->rendererPtr.get());
    CHECK(renderer != nullptr);

    QueryFramebufferUqPtr& framebufferPtr = itr->second.framebufferPtr;
    framebufferPtr->bindToRenderer(renderer, FboBind::READ);
    pixelsPtr = framebufferPtr->readColorBuffer(0, 0, width, height);

    renderer->makeInactive();
  }

  return PngData(width, height, pixelsPtr, compressionLevel);
}

unsigned int QueryRenderer::getIdAt(int x, int y) {
  // RUNTIME_EX_ASSERT(_framebufferPtr != nullptr,
  //                   "QueryRenderer: The framebuffer is not defined. Cannot retrieve id at pixel.");

  // // TODO(croot): develop an API for reading from specific fbo buffers
  // _framebufferPtr->bindToRenderer();
  // MAPD_CHECK_GL_ERROR(glReadBuffer(GL_COLOR_ATTACHMENT1));

  // // TODO(croot): support a wider pixel check for a hit test and take a weighted avg
  // // of the results to get a more stable result at boundaries
  // unsigned int id;
  // MAPD_CHECK_GL_ERROR(glReadPixels(int(x), int(y), 1, 1, GL_RED_INTEGER, GL_UNSIGNED_INT, &id));

  std::shared_ptr<unsigned int> idPixelsPtr;

  if (_compositorPtr) {
    Renderer* renderer = _compositorPtr->getRenderer();
    renderer->makeActiveOnCurrentThread();
    idPixelsPtr = _compositorPtr->readIdBuffer(x, y, 1, 1);
    renderer->makeInactive();
  } else {
    auto itr = _perGpuData.begin();

    QueryRenderManager::PerGpuDataShPtr qrmGpuData = itr->second.getQRMGpuData();

    // const GpuId& gpuId = itr->first;
    itr->second.makeActiveOnCurrentThread();
    GLRenderer* renderer = dynamic_cast<GLRenderer*>(qrmGpuData->rendererPtr.get());
    CHECK(renderer != nullptr);

    QueryFramebufferUqPtr& framebufferPtr = itr->second.framebufferPtr;
    framebufferPtr->bindToRenderer(renderer, FboBind::READ);
    idPixelsPtr = framebufferPtr->readIdBuffer(x, y, 1, 1);
    itr->second.makeInactive();
  }

  CHECK(idPixelsPtr);

  return idPixelsPtr.get()[0];
}

QueryRendererContext::QueryRendererContext(bool doHitTest, bool doDepthTest)
    : executor_(nullptr),
      _perGpuData(),
      _width(0),
      _height(0),
      _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _invalidKey(std::numeric_limits<int64_t>::max()),
      _jsonCache(nullptr) {
}

QueryRendererContext::QueryRendererContext(int width, int height, bool doHitTest, bool doDepthTest)
    : executor_(nullptr),
      _perGpuData(),
      _width(width),
      _height(height),
      _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _invalidKey(std::numeric_limits<int64_t>::max()),
      _jsonCache(nullptr),
      _queryDataLayoutPtr(nullptr) {
}

QueryRendererContext::~QueryRendererContext() {
  _clear();
  _clearGpuResources();
}

void QueryRendererContext::_clear() {
  _width = 0;
  _height = 0;
  _dataTableMap.clear();
  _scaleConfigMap.clear();
  _geomConfigs.clear();
  _eventCallbacksMap.clear();
}

void QueryRendererContext::_clearGpuResources() {
  _perGpuData.clear();
}

void QueryRendererContext::_initGpuResources(QueryRenderer::PerGpuDataMap& qrPerGpuData,
                                             const std::unordered_set<GpuId>& unusedGpus) {
  // CHECK(_perGpuData.size() == qrmPerGpuData.size());
  // for (size_t i = 0; i < qrmPerGpuData.size(); ++i) {
  //   _perGpuData[i].qrmGpuData = &qrmPerGpuData[i];
  // }

  for (const auto& itr : qrPerGpuData) {
    if (_perGpuData.find(itr.first) == _perGpuData.end()) {
      _perGpuData.insert({itr.first, PerGpuData(itr.second.qrmGpuData)});
    }
  }

  for (auto gpuId : unusedGpus) {
    _perGpuData.erase(gpuId);
  }
}

void QueryRendererContext::_updateConfigGpuResources(const std::unordered_set<GpuId>& unusedGpus) {
  for (auto dataItr : _dataTableMap) {
    dataItr.second->_initGpuResources(this, unusedGpus, false);
  }

  // for (auto scaleItr : _scaleConfigMap) {
  //   scaleItr.second->_initGpuResources(qrPerGpuData, unusedGpus);
  // }

  for (auto geomItr : _geomConfigs) {
    geomItr->_initGpuResources(this, unusedGpus, false);
  }
}

bool QueryRendererContext::hasDataTable(const std::string& tableName) const {
  return (_dataTableMap.find(tableName) != _dataTableMap.end());
}

QueryDataTableVBOShPtr QueryRendererContext::getDataTable(const std::string& tableName) const {
  QueryDataTableVBOShPtr rtn(nullptr);

  auto itr = _dataTableMap.find(tableName);
  if (itr != _dataTableMap.end()) {
    rtn = itr->second;
  }

  return rtn;
}

bool QueryRendererContext::hasScale(const std::string& scaleConfigName) const {
  return (_scaleConfigMap.find(scaleConfigName) != _scaleConfigMap.end());
}

ScaleShPtr QueryRendererContext::getScale(const std::string& scaleConfigName) const {
  ScaleShPtr rtn(nullptr);

  auto itr = _scaleConfigMap.find(scaleConfigName);
  if (itr != _scaleConfigMap.end()) {
    rtn = itr->second;
  }

  return rtn;
}

QueryResultVertexBufferShPtr QueryRendererContext::getQueryResultVertexBuffer(const GpuId& gpuId) const {
  // TODO(croot): make thread safe?

  PerGpuDataMap::const_iterator itr = _perGpuData.find(gpuId);

  RUNTIME_EX_ASSERT(itr != _perGpuData.end(),
                    "Cannot get query result vertex buffer for gpuId " + std::to_string(gpuId) + ".");
  QueryRenderManager::PerGpuDataShPtr qrmGpuData = itr->second.getQRMGpuData();
  CHECK(qrmGpuData);
  return qrmGpuData->queryResultBufferPtr;
}

std::map<GpuId, QueryVertexBufferShPtr> QueryRendererContext::getQueryResultVertexBuffers() const {
  std::map<GpuId, QueryVertexBufferShPtr> rtn;

  QueryRenderManager::PerGpuDataShPtr qrmGpuData;
  for (auto& itr : _perGpuData) {
    // rtn.insert({itr.first,
    // std::static_pointer_cast<QueryVertexBuffer>(itr.second.qrmGpuData->queryResultBufferPtr)});
    qrmGpuData = itr.second.getQRMGpuData();
    CHECK(qrmGpuData);
    rtn.insert({itr.first, qrmGpuData->queryResultBufferPtr});
  }

  return rtn;
}

bool QueryRendererContext::isJSONCacheUpToDate(const rapidjson::Pointer& objPath, const rapidjson::Value& obj) {
  if (!_jsonCache) {
    return false;
  }

  const rapidjson::Value* cachedVal = GetValueByPointer(*_jsonCache, objPath);

  // TODO(croot): throw an exception or just return false?
  RUNTIME_EX_ASSERT(cachedVal != nullptr,
                    "The path " + RapidJSONUtils::getPointerPath(objPath) + " is not a valid path in the cached json.");

  return (cachedVal ? (*cachedVal == obj) : false);
}

void QueryRendererContext::subscribeToRefEvent(RefEventType eventType,
                                               const ScaleShPtr& eventObj,
                                               RefEventCallback cb) {
  const std::string& eventObjName = eventObj->getNameRef();
  RUNTIME_EX_ASSERT(
      hasScale(eventObjName),
      "Cannot subscribe to event for scale \"" + eventObj->getNameRef() + "\". The scale does not exist.");

  EventCallbacksMap::iterator itr;

  if ((itr = _eventCallbacksMap.find(eventObjName)) == _eventCallbacksMap.end()) {
    itr = _eventCallbacksMap.insert(itr, std::make_pair(eventObjName, std::move(EventCallbacksArray())));
  }

  size_t idx = static_cast<size_t>(eventType);
  if (eventType == RefEventType::ALL) {
    for (size_t i = 0; i < idx; ++i) {
      itr->second[i].insert(cb);
    }
  } else {
    itr->second[idx].insert(cb);
  }
}

void QueryRendererContext::unsubscribeFromRefEvent(RefEventType eventType,
                                                   const ScaleShPtr& eventObj,
                                                   RefEventCallback cb) {
  EventCallbacksMap::iterator mitr;
  EventCallbackList::iterator sitr;
  const std::string& eventObjName = eventObj->getNameRef();

  if ((mitr = _eventCallbacksMap.find(eventObjName)) != _eventCallbacksMap.end()) {
    size_t idx = static_cast<size_t>(eventType);

    if (eventType == RefEventType::ALL) {
      for (size_t i = 0; i < idx; ++i) {
        if ((sitr = mitr->second[i].find(cb)) != mitr->second[i].end()) {
          mitr->second[i].erase(sitr);
        }
      }
    } else {
      if ((sitr = mitr->second[idx].find(cb)) != mitr->second[idx].end()) {
        mitr->second[idx].erase(sitr);
      }
    }
  }

  // TODO(croot): throw an error or warning?
}

void QueryRendererContext::_fireRefEvent(RefEventType eventType, const ScaleShPtr& eventObj) {
  CHECK(eventType != RefEventType::ALL);

  EventCallbacksMap::iterator mitr;
  const std::string& eventObjName = eventObj->getNameRef();

  if ((mitr = _eventCallbacksMap.find(eventObjName)) != _eventCallbacksMap.end()) {
    size_t idx = static_cast<size_t>(eventType);

    std::vector<RefEventCallback> callbacksToCall(mitr->second[idx].size());

    int i = 0;
    for (auto& cb : mitr->second[idx]) {
      // callbacks have the ability to subscribe and unsubscribe from the events
      // so we can't just call them here while looping through the data structure
      // that holds the callbacks as that data structure can be modified mid-stream.
      // So we'll store an additional data structure for all callbacks that need
      // calling and call them.
      callbacksToCall[i++] = cb;
    }

    for (auto& cb : callbacksToCall) {
      cb(eventType, eventObj);
    }
  }

  // TODO(croot): throw an error or warning if eventObj not found in map?
}

void QueryRendererContext::_update() {
  for (auto& geomConfig : _geomConfigs) {
    geomConfig->update();
  }
}

// void QueryRenderer::_buildShaderFromGeomConfig(const GeomConfigPtr& geomConfigPtr) {

// }

std::string getDataTableNameFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "A data object in the JSON must be an object.");

  rapidjson::Value::ConstMemberIterator itr;

  RUNTIME_EX_ASSERT((itr = obj.FindMember("name")) != obj.MemberEnd() && itr->value.IsString(),
                    "A data object must contain a \"name\" property and it must be a string");

  return itr->value.GetString();
}

QueryDataTableType getDataTableTypeFromJSONObj(const rapidjson::Value& obj) {
  RUNTIME_EX_ASSERT(obj.IsObject(), "A data table in the JSON must be an object.");

  rapidjson::Value::ConstMemberIterator itr;

  if ((itr = obj.FindMember("sql")) != obj.MemberEnd()) {
    RUNTIME_EX_ASSERT(itr->value.IsString(),
                      "Cannot get data table's type - the sql property for a data table must be a string.");
    return QueryDataTableType::SQLQUERY;
  } else if ((itr = obj.FindMember("values")) != obj.MemberEnd()) {
    return QueryDataTableType::EMBEDDED;
  } else if ((itr = obj.FindMember("url")) != obj.MemberEnd()) {
    return QueryDataTableType::URL;
  }

  THROW_RUNTIME_EX("Cannot get data table's type - the data table's type is not supported.");
  return QueryDataTableType::UNSUPPORTED;
}

QueryDataTableVBOShPtr createDataTable(const rapidjson::Value& obj,
                                       const rapidjson::Pointer& objPath,
                                       const QueryRendererContextShPtr& ctx,
                                       const std::string& name) {
  std::string tableName(name);
  if (!tableName.length()) {
    tableName = getDataTableNameFromJSONObj(obj);
  } else {
    RUNTIME_EX_ASSERT(obj.IsObject(), "Cannot create data table - A data object in the JSON must be an object.");
  }

  RUNTIME_EX_ASSERT(tableName.length(),
                    "Cannot create data table - The data table has an empty name. It must have a name.");

  QueryDataTableType tableType = getDataTableTypeFromJSONObj(obj);
  switch (tableType) {
    case QueryDataTableType::SQLQUERY:
      return QueryDataTableVBOShPtr(
          new SqlQueryDataTable(ctx, tableName, obj, objPath, ctx->getQueryResultVertexBuffers()));
    case QueryDataTableType::EMBEDDED:
    case QueryDataTableType::URL:
      return QueryDataTableVBOShPtr(
          new DataTable(ctx, tableName, obj, objPath, tableType, ctx->doHitTest(), DataTable::VboType::INTERLEAVED));
    default:
      THROW_RUNTIME_EX(
          "Cannot create data table \"" + tableName +
          "\". It is not a supported table. Supported tables must have an \"sql\", \"values\" or \"url\" property.");
  }

  return QueryDataTableVBOShPtr(nullptr);
}

}  // namespace QueryRenderer
