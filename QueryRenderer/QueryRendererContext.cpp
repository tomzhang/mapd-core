#include "QueryRenderer.h"
#include "Data/BaseQueryDataTable.h"
#include "Marks/BaseMark.h"
#include "Scales/Scale.h"
#include "Rendering/QueryFramebuffer.h"
#include "Rendering/QueryRenderCompositor.h"
#include <Rendering/Window.h>
#include <Rendering/Renderer/GL/GLRenderer.h>

#include <png.h>
#include <vector>
#include <thread>
#include <cstring>
#include "rapidjson/error/en.h"

namespace QueryRenderer {

QueryRendererContext::QueryRendererContext(int userId, int widgetId, bool doHitTest, bool doDepthTest)
    : executor_(nullptr),
      _perGpuData(),
      _userWidget(userId, widgetId),
      _width(0),
      _height(0),
      _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _invalidKey(std::numeric_limits<int64_t>::max()),
      _jsonCache(nullptr) {
}

QueryRendererContext::QueryRendererContext(int userId,
                                           int widgetId,
                                           int width,
                                           int height,
                                           bool doHitTest,
                                           bool doDepthTest)
    : executor_(nullptr),
      _perGpuData(),
      _userWidget(userId, widgetId),
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

void QueryRendererContext::_clear(bool preserveDimensions) {
  if (!preserveDimensions) {
    _width = 0;
    _height = 0;
  }
  _dataTableMap.clear();
  _scaleConfigMap.clear();
  _geomConfigs.clear();
  _eventCallbacksMap.clear();
  _jsonCache = nullptr;
}

void QueryRendererContext::_clearGpuResources() {
  _perGpuData.clear();
}

void QueryRendererContext::_initGpuResources(QueryRenderer::PerGpuDataMap& qrPerGpuData,
                                             const std::unordered_set<GpuId>& unusedGpus) {
  for (const auto& itr : qrPerGpuData) {
    if (_perGpuData.find(itr.first) == _perGpuData.end()) {
      _perGpuData.insert({itr.first, PerGpuData(itr.second)});
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

QueryDataTableShPtr QueryRendererContext::getDataTable(const std::string& tableName) const {
  QueryDataTableShPtr rtn(nullptr);

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
                    "QueryRendererContext " + to_string(_userWidget) +
                        ": Cannot get query result vertex buffer for gpuId " + std::to_string(gpuId) + ".");
  RootPerGpuDataShPtr qrmGpuData = itr->second.getRootPerGpuData();
  CHECK(qrmGpuData);
  return qrmGpuData->queryResultBufferPtr;
}

std::map<GpuId, QueryVertexBufferShPtr> QueryRendererContext::getQueryResultVertexBuffers() const {
  std::map<GpuId, QueryVertexBufferShPtr> rtn;

  RootPerGpuDataShPtr qrmGpuData;
  for (auto& itr : _perGpuData) {
    qrmGpuData = itr.second.getRootPerGpuData();
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
                    "QueryRendererContext " + to_string(_userWidget) + ": The path " +
                        RapidJSONUtils::getPointerPath(objPath) + " is not a valid path in the cached json.");

  return (cachedVal ? (*cachedVal == obj) : false);
}

void QueryRendererContext::subscribeToRefEvent(RefEventType eventType,
                                               const ScaleShPtr& eventObj,
                                               RefEventCallback cb) {
  const std::string& eventObjName = eventObj->getNameRef();
  RUNTIME_EX_ASSERT(hasScale(eventObjName),
                    "QueryRendererContext " + to_string(_userWidget) + ": Cannot subscribe to event for scale \"" +
                        eventObj->getNameRef() + "\". The scale does not exist.");

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

}  // namespace QueryRenderer
