#include "QueryRenderer.h"
#include "QueryFramebuffer.h"
#include "QueryDataTable.h"
#include "QueryRendererObjects.h"
#include <Rendering/Renderer/GL/GLRenderer.h>

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

using ::Rendering::GL::GLRenderer;

QueryRenderer::QueryRenderer(bool doHitTest, bool doDepthTest)
    : _ctx(new QueryRendererContext(doHitTest, doDepthTest)), _perGpuData() {
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

void QueryRenderer::_initGpuResources(const std::vector<GpuId>& gpuIds,
                                      QueryRenderManager::PerGpuDataMap& qrmPerGpuData) {
  std::unordered_set<GpuId> unusedGpus;
  unusedGpus.reserve(_perGpuData.size());
  for (const auto& kv : _perGpuData) {
    unusedGpus.insert(kv.first);
  }

  for (auto gpuId : gpuIds) {
    auto myItr = _perGpuData.find(gpuId);
    if (myItr == _perGpuData.end()) {
      auto itr = qrmPerGpuData.find(gpuId);
      CHECK(itr != qrmPerGpuData.end());

      PerGpuData gpuData;

      gpuData.qrmGpuData = &(itr->second);

      // TODO(croot): validate the QueryRenderManager data is complete?
      CHECK(itr->second.rendererPtr != nullptr);

      _perGpuData.emplace(gpuId, std::move(gpuData));
    } else {
      myItr->second.framebufferPtr->resize(_ctx->getWidth(), _ctx->getHeight());

      unusedGpus.erase(gpuId);
    }
  }

  // now clean up any unused gpu resources
  for (auto gpuId : unusedGpus) {
    _perGpuData.erase(gpuId);
  }

  _ctx->_initGpuResources(_perGpuData, unusedGpus);
}

void QueryRenderer::_resizeFramebuffers(int width, int height) {
  for (auto& gpuDataItr : _perGpuData) {
    if (gpuDataItr.second.framebufferPtr == nullptr) {
      CHECK(gpuDataItr.second.qrmGpuData->rendererPtr != nullptr);
      GLRenderer* renderer = dynamic_cast<GLRenderer*>(gpuDataItr.second.qrmGpuData->rendererPtr.get());
      CHECK(renderer != nullptr);

      gpuDataItr.second.framebufferPtr.reset(
          new QueryFramebuffer(renderer, width, height, _ctx->doHitTest(), _ctx->doDepthTest()));
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

void QueryRenderer::updateQueryResultBufferPostQuery(QueryDataLayout* dataLayoutPtr,
                                                     QueryRenderManager::PerGpuDataMap& qrmPerGpuData) {
  std::vector<GpuId> gpuIds;
  for (auto& item : dataLayoutPtr->numRowsPerGpuBufferMap) {
    gpuIds.push_back(item.first);
  }
  _initGpuResources(gpuIds, qrmPerGpuData);
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

void QueryRenderer::activateGpu(const GpuId& gpuId, QueryRenderManager::PerGpuDataMap& qrmPerGpuData) {
  _initGpuResources({gpuId}, qrmPerGpuData);
}

void QueryRenderer::render() {
  // RUNTIME_EX_ASSERT(_framebufferPtr != nullptr, "QueryRenderer: The framebuffer is not defined. Cannot render.");

  // _framebufferPtr->bindToRenderer();

  // MAPD_CHECK_GL_ERROR(glEnable(GL_BLEND));
  // MAPD_CHECK_GL_ERROR(glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD));
  // // MAPD_CHECK_GL_ERROR(glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO));
  // // MAPD_CHECK_GL_ERROR(glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE));
  // MAPD_CHECK_GL_ERROR(glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA));

  // MAPD_CHECK_GL_ERROR(glClearColor(0, 0, 0, 1));
  // MAPD_CHECK_GL_ERROR(glViewport(0, 0, _ctx->_width, _ctx->_height));
  // MAPD_CHECK_GL_ERROR(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  // for (size_t i = 0; i < _ctx->_geomConfigs.size(); ++i) {
  //   _ctx->_geomConfigs[i]->draw();
  // }
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

  // return id;
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
  return itr->second.qrmGpuData->queryResultBufferPtr;
}

std::map<GpuId, QueryVertexBufferShPtr> QueryRendererContext::getQueryResultVertexBuffers() const {
  std::map<GpuId, QueryVertexBufferShPtr> rtn;

  for (auto& itr : _perGpuData) {
    // rtn.insert({itr.first,
    // std::static_pointer_cast<QueryVertexBuffer>(itr.second.qrmGpuData->queryResultBufferPtr)});
    rtn.insert({itr.first, itr.second.qrmGpuData->queryResultBufferPtr});
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
