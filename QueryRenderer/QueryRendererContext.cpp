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

QueryRendererContext::QueryRendererContext(int userId,
                                           int widgetId,
                                           const std::shared_ptr<RootCache>& qrmGpuCache,
                                           bool doHitTest,
                                           bool doDepthTest)
    : _qrmGpuCache(qrmGpuCache),
      executor_(nullptr),
      _userWidget(userId, widgetId),
      _width(0),
      _height(0),
      _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _jsonCache(nullptr),
      _currCbId(0) {
}

QueryRendererContext::QueryRendererContext(int userId,
                                           int widgetId,
                                           const std::shared_ptr<RootCache>& qrmGpuCache,
                                           int width,
                                           int height,
                                           bool doHitTest,
                                           bool doDepthTest)
    : _qrmGpuCache(qrmGpuCache),
      executor_(nullptr),
      _userWidget(userId, widgetId),
      _width(width),
      _height(height),
      _doHitTest(doHitTest),
      _doDepthTest(doDepthTest),
      _jsonCache(nullptr),
      _currCbId(0),
      _queryDataLayoutPtr(nullptr) {
}

QueryRendererContext::~QueryRendererContext() {
  _clear();
  // _clearGpuResources();
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

void QueryRendererContext::_updateConfigGpuResources() {
  auto qrmGpuCache = getRootGpuCache();
  CHECK(qrmGpuCache);

  for (auto dataPtr : _dataTableMap) {
    auto dtPtr = std::dynamic_pointer_cast<BaseQueryDataTable>(dataPtr);
    CHECK(dtPtr);
    dtPtr->_initGpuResources(qrmGpuCache);
  }

  for (auto& scalePtr : _scaleConfigMap) {
    scalePtr->_initGpuResources(this, false);
  }

  for (auto geomItr : _geomConfigs) {
    geomItr->_initGpuResources(this, false);
  }
}

bool QueryRendererContext::hasDataTable(const std::string& tableName) const {
  auto& nameLookup = _dataTableMap.get<DataTableName>();
  return (nameLookup.find(tableName) != nameLookup.end());
}

QueryDataTableShPtr QueryRendererContext::getDataTable(const std::string& tableName) const {
  QueryDataTableShPtr rtn(nullptr);

  auto& nameLookup = _dataTableMap.get<DataTableName>();
  auto itr = nameLookup.find(tableName);
  if (itr != nameLookup.end()) {
    rtn = std::dynamic_pointer_cast<BaseQueryDataTable>(*itr);
    CHECK(rtn);
  }

  return rtn;
}

bool QueryRendererContext::hasScale(const std::string& scaleConfigName) const {
  auto& nameLookup = _scaleConfigMap.get<ScaleName>();
  return (nameLookup.find(scaleConfigName) != nameLookup.end());
}

ScaleShPtr QueryRendererContext::getScale(const std::string& scaleConfigName) const {
  ScaleShPtr rtn(nullptr);

  auto& nameLookup = _scaleConfigMap.get<ScaleName>();
  auto itr = nameLookup.find(scaleConfigName);
  if (itr != nameLookup.end()) {
    rtn = *itr;
  }

  return rtn;
}

bool QueryRendererContext::isJSONCacheUpToDate(const rapidjson::Pointer& objPath, const rapidjson::Value& obj) {
  if (!_jsonCache) {
    return false;
  }

  const rapidjson::Value* cachedVal = GetValueByPointer(*_jsonCache, objPath);

  return (cachedVal ? (*cachedVal == obj) : false);
}

const rapidjson::Value* QueryRendererContext::getJSONObj(const rapidjson::Pointer& objPath) {
  if (_jsonCache) {
    return GetValueByPointer(*_jsonCache, objPath);
  }

  return nullptr;
}

RefCallbackId QueryRendererContext::subscribeToRefEvent(RefEventType eventType,
                                                        const RefObjShPtr& eventObj,
                                                        RefEventCallback cb) {
  auto refType = eventObj->getRefType();
  const std::string& eventObjName = eventObj->getNameRef();

  switch (refType) {
    case RefType::DATA: {
      RUNTIME_EX_ASSERT(hasDataTable(eventObjName),
                        "QueryRendererContext " + to_string(_userWidget) +
                            ": Cannot subscribe to event for data table \"" + eventObj->getNameRef() +
                            "\". The data table does not exist.");
    } break;
    case RefType::SCALE:
      RUNTIME_EX_ASSERT(hasScale(eventObjName),
                        "QueryRendererContext " + to_string(_userWidget) + ": Cannot subscribe to event for scale \"" +
                            eventObj->getNameRef() + "\". The scale does not exist.");
      break;
    default:
      THROW_RUNTIME_EX("QueryRendererContext " + to_string(_userWidget) +
                       ": Cannot subscribe to event for an object of type " + to_string(refType) + ". " +
                       to_string(refType) + " is an unsupported type.");
  }

  auto refTypeConv = static_cast<int>(refType);
  auto id = ++_currCbId;

  EventCallbacksByNameMap::iterator nameItr;
  EventCallbacksMap::iterator typeItr;

  if ((typeItr = _eventCallbacksMap.find(refTypeConv)) == _eventCallbacksMap.end()) {
    EventCallbacksByNameMap nameMap = {{eventObjName, std::move(EventCallbacksArray())}};
    typeItr = _eventCallbacksMap.insert(typeItr, std::make_pair(refTypeConv, std::move(nameMap)));
    nameItr = typeItr->second.begin();
  } else if ((nameItr = typeItr->second.find(eventObjName)) == typeItr->second.end()) {
    nameItr = typeItr->second.insert(nameItr, std::make_pair(eventObjName, std::move(EventCallbacksArray())));
  }

  size_t idx = static_cast<size_t>(eventType);
  if (eventType == RefEventType::ALL) {
    for (size_t i = 0; i < idx; ++i) {
      CHECK(nameItr->second[i].emplace(id, cb).second);
    }
  } else {
    CHECK(nameItr->second[idx].emplace(id, cb).second);
  }

  return id;
}

void QueryRendererContext::unsubscribeFromRefEvent(RefEventType eventType,
                                                   const RefObjShPtr& eventObj,
                                                   const RefCallbackId cbId) {
  EventCallbacksByNameMap::iterator nitr;
  EventCallbacksMap::iterator mitr;
  EventCallbackMap::iterator sitr;

  const std::string& eventObjName = eventObj->getNameRef();
  auto refType = static_cast<int>(eventObj->getRefType());

  if ((mitr = _eventCallbacksMap.find(refType)) != _eventCallbacksMap.end() &&
      (nitr = mitr->second.find(eventObjName)) != mitr->second.end()) {
    size_t idx = static_cast<size_t>(eventType);

    if (eventType == RefEventType::ALL) {
      for (size_t i = 0; i < idx; ++i) {
        if ((sitr = nitr->second[i].find(cbId)) != nitr->second[i].end()) {
          nitr->second[i].erase(sitr);
        }
      }
    } else {
      if ((sitr = nitr->second[idx].find(cbId)) != nitr->second[idx].end()) {
        nitr->second[idx].erase(sitr);
      }
    }

    if (!nitr->second.size()) {
      mitr->second.erase(eventObjName);
    }
    if (!mitr->second.size()) {
      _eventCallbacksMap.erase(refType);
    }
  }

  // TODO(croot): throw an error or warning?
}

std::set<GpuId> QueryRendererContext::getUsedGpus() const {
  std::set<GpuId> rtn;

  for (auto& geomConfig : _geomConfigs) {
    for (auto gpuId : geomConfig->getUsedGpus()) {
      rtn.insert(gpuId);
    }
  }

  return rtn;
}

void QueryRendererContext::addAccumulatorScale(const std::string& scaleName) {
  _accumulatorScales.insert(scaleName);
}

void QueryRendererContext::removeAccumulatorScale(const std::string& scaleName) {
  _accumulatorScales.erase(scaleName);
}

void QueryRendererContext::_fireRefEvent(RefEventType eventType, const RefObjShPtr& eventObj) {
  CHECK(eventType != RefEventType::ALL);

  EventCallbacksByNameMap::iterator nitr;
  EventCallbacksMap::iterator mitr;

  auto refType = static_cast<int>(eventObj->getRefType());
  auto& eventObjName = eventObj->getNameRef();

  if ((mitr = _eventCallbacksMap.find(refType)) != _eventCallbacksMap.end() &&
      (nitr = mitr->second.find(eventObjName)) != mitr->second.end()) {
    size_t idx = static_cast<size_t>(eventType);

    std::vector<RefEventCallback> callbacksToCall(nitr->second[idx].size());

    int i = 0;
    for (auto& cb : nitr->second[idx]) {
      // callbacks have the ability to subscribe and unsubscribe from the events
      // so we can't just call them here while looping through the data structure
      // that holds the callbacks as that data structure can be modified mid-stream.
      // So we'll store an additional data structure for all callbacks that need
      // calling and call them.
      callbacksToCall[i++] = cb.second;
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
