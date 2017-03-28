#include "QueryRenderer.h"
#include "QueryRendererContext.h"
#include "Data/QueryDataTable.h"
#include "Data/Utils.h"
#include "Marks/BaseMark.h"
#include "Marks/Utils.h"
#include "Scales/Scale.h"
#include "Scales/Utils.h"
#include "Rendering/QueryFramebuffer.h"
#include "Rendering/QueryRenderCompositor.h"
#include "Rendering/QueryRenderSMAAPass.h"
#include "Rendering/DistributedRenderBufferCompositor.h"
#include <Rendering/Window.h>
#include <Rendering/Renderer/GL/GLRenderer.h>

#include <Rendering/Renderer/GL/Resources/GLTexture2dArray.h>

#include <png.h>
#include <vector>
#include <thread>
#include <cstring>
#include "rapidjson/error/en.h"
#include <boost/algorithm/string/join.hpp>

namespace QueryRenderer {

using ::Rendering::Renderer;
using ::Rendering::WindowShPtr;
using ::Rendering::RendererShPtr;
using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::Resources::FboBind;

static const GpuId EMPTY_GPUID = 1000000000;

typedef ::Rendering::Objects::Array2d<float> Array2df;

static Array2df createGaussianKernel(int kernelSize, float stddev = 1.0) {
  // NOTE: it is reasonable to create the gaussian kernel as a 2D
  // array for getIdAt() since we are only using the gaussian values
  // for sums and not as coefficients. If the latter, like in performing
  // a gaussian blur in image processing, then we'd want to take advantage
  // of the fact that the gaussian kernel is seperable:
  // See:
  // https://en.wikipedia.org/wiki/Gaussian_blur
  // and
  // https://en.wikipedia.org/wiki/Separable_filter
  assert(kernelSize > 0 && kernelSize % 2 == 1);

  Array2df kernel(kernelSize, kernelSize);

  std::unique_ptr<float[]> kernel1d(new float[kernelSize]);
  int i, j;
  float stddevsq2x = 2.0 * stddev * stddev;

  // NOTE: kernel size will always be odd
  int halfKernelSize = kernelSize / 2;
  for (i = -halfKernelSize, j = 0; i <= halfKernelSize; ++i, ++j) {
    kernel1d[j] = std::pow(M_E, -float(i * i) / stddevsq2x) / std::sqrt(M_PI * stddevsq2x);
  }

  for (i = 0; i < kernelSize; ++i) {
    for (j = 0; j < kernelSize; ++j) {
      kernel[i][j] = kernel1d[i] * kernel1d[j];
    }
  }

  return kernel;
}

static bool idCounterCompare(const std::pair<unsigned int, float>& a, const std::pair<unsigned int, float>& b) {
  return (a.second < b.second);
}

QueryRenderer::QueryRenderer(int userId,
                             int widgetId,
                             const std::shared_ptr<RootCache>& qrmGpuCache,
                             bool doHitTest,
                             bool doDepthTest) noexcept
    : _ctx(new QueryRendererContext(userId, widgetId, qrmGpuCache, doHitTest, doDepthTest)),
      _pboGpu(EMPTY_GPUID),
      _idPixelsDirty(false) {}

QueryRenderer::QueryRenderer(int userId,
                             int widgetId,
                             const std::shared_ptr<RootCache>& qrmGpuCache,
                             const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr,
                             bool doHitTest,
                             bool doDepthTest)
    : QueryRenderer(userId, widgetId, qrmGpuCache, doHitTest, doDepthTest) {
  _initFromJSON(jsonDocumentPtr);
}

QueryRenderer::QueryRenderer(int userId,
                             int widgetId,
                             const std::shared_ptr<RootCache>& qrmGpuCache,
                             const std::string& configJSON,
                             bool doHitTest,
                             bool doDepthTest)
    : QueryRenderer(userId, widgetId, qrmGpuCache, doHitTest, doDepthTest) {
  _initFromJSON(configJSON);
}

QueryRenderer::~QueryRenderer() {
  _clearAll();
}

void QueryRenderer::_clear(bool preserveDimensions) {
  _ctx->_clear(preserveDimensions);
}

void QueryRenderer::_clearGpuResources() {
  _releasePbo();
}

void QueryRenderer::_clearAll(bool preserveDimensions) {
  _clear(preserveDimensions);
  _clearGpuResources();
}

void QueryRenderer::_resizeFramebuffers(int width, int height) {
  auto qrmGpuCache = _ctx->getRootGpuCache();
  CHECK(qrmGpuCache);
  auto qrmPerGpuDataPtr = qrmGpuCache->perGpuData;
  CHECK(qrmPerGpuDataPtr != nullptr);

  // for EGL compositing to work, the compositor needs to be
  // resized last, so we'll force the resize of the non-compositor
  // framebuffers first.
  // We're keeping this logic for the other window manager
  // libraries (i.e. GLX) as it will work there too.

  // get the iterator to the compositor gpu first
  if (qrmPerGpuDataPtr->size() > 1) {
    auto qrmItr = qrmPerGpuDataPtr->begin();
    qrmItr = qrmPerGpuDataPtr->find((*qrmItr)->getCompositorGpuId());
    CHECK(qrmItr != qrmPerGpuDataPtr->end());

    // resize the non-compositor gpu framebuffers first
    for (auto& gpuDataItr : (*qrmPerGpuDataPtr)) {
      if (gpuDataItr->gpuId != (*qrmItr)->gpuId) {
        gpuDataItr->makeActiveOnCurrentThread();
        gpuDataItr->resize(width, height, false);
      }
    }

    // now resize the compositor gpu framebuffers, make sure to resize
    // the scale accumulation textures before the compositor too.
    (*qrmItr)->makeActiveOnCurrentThread();
    (*qrmItr)->resize(width, height, true);
  } else {
    for (auto& gpuDataItr : (*qrmPerGpuDataPtr)) {
      gpuDataItr->makeActiveOnCurrentThread();
      gpuDataItr->resize(width, height);
    }
  }
}

void QueryRenderer::_initFromJSON(const std::string& configJSON, bool forceUpdate) {
  std::shared_ptr<rapidjson::Document> objPtr(new rapidjson::Document());

  objPtr->Parse(configJSON.c_str());

  // TODO(croot): this can be removed if the executor will handle the initial parse.
  RUNTIME_EX_ASSERT(
      !objPtr->HasParseError(),
      RapidJSONUtils::getJsonParseErrorStr("json offset: " + std::to_string(objPtr->GetErrorOffset()) + ", error: " +
                                           rapidjson::GetParseError_En(objPtr->GetParseError())));

  _initFromJSON(objPtr);
}

void QueryRenderer::_initFromJSON(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate) {
  rapidjson::Pointer rootPath;
  rapidjson::Value* obj = jsonDocumentPtr.get();

  if (forceUpdate) {
    _clear();
  }

  if (!_ctx->_jsonCache || *_ctx->_jsonCache != *jsonDocumentPtr) {
    RUNTIME_EX_ASSERT(
        obj->IsObject(),
        RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), *obj, "Root object is not a JSON object."));

    LOG(INFO) << "Render initFromJSON " << _ctx->getUserWidgetIds()
              << ", json: " << RapidJSONUtils::getObjAsString(*obj);

    rapidjson::Value::ConstMemberIterator mitr;
    rapidjson::Value::ConstValueIterator vitr;

    RUNTIME_EX_ASSERT(
        (mitr = obj->FindMember("width")) != obj->MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), *obj, "\"width\" is not defined."));
    RUNTIME_EX_ASSERT(mitr->value.IsUint(),
                      RapidJSONUtils::getJsonParseErrorStr(
                          _ctx->getUserWidgetIds(), mitr->value, "\"width\" is not an unsigned integer."));
    size_t width = mitr->value.GetUint();

    RUNTIME_EX_ASSERT(
        (mitr = obj->FindMember("height")) != obj->MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), *obj, "\"height\" is not defined."));
    RUNTIME_EX_ASSERT(mitr->value.IsUint(),
                      RapidJSONUtils::getJsonParseErrorStr(
                          _ctx->getUserWidgetIds(), mitr->value, "\"height\" is not an unsigned integer."));
    size_t height = mitr->value.GetUint();

    setWidthHeight(width, height);

    std::array<std::vector<QueryDataTableJSONShPtr>, static_cast<size_t>(RefEventType::ALL)> dataEvents;
    std::string propName = "data";
    {
      mitr = obj->FindMember(propName.c_str());
      if (mitr != obj->MemberEnd()) {
        rapidjson::Pointer dataPath = rootPath.Append(propName.c_str(), propName.length());

        RUNTIME_EX_ASSERT(
            mitr->value.IsArray(),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), mitr->value, "the \"" + propName + "\" member must be an array."));

        QueryDataTableShPtr dataTablePtr;
        std::unordered_set<std::string> visitedNames;
        std::unordered_set<std::string> unvisitedNames;
        unvisitedNames.reserve(_ctx->_dataTableMap.size());
        for (auto kv : _ctx->_dataTableMap) {
          unvisitedNames.insert(kv->getNameRef());
        }

        for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
          rapidjson::Pointer dataObjPath = dataPath.Append(vitr - mitr->value.Begin());

          std::string tableName = getDataTableNameFromJSONObj(*vitr);

          RUNTIME_EX_ASSERT(
              visitedNames.find(tableName) == visitedNames.end(),
              RapidJSONUtils::getJsonParseErrorStr(
                  _ctx->getUserWidgetIds(), *vitr, "a data table with the name \"" + tableName + "\" already exists."));

          dataTablePtr = _ctx->getDataTable(tableName);

          if (!dataTablePtr) {
            dataTablePtr = createDataTable(*vitr, dataObjPath, _ctx, tableName);
            auto dtPtr = std::dynamic_pointer_cast<BaseQueryDataTableJSON>(dataTablePtr);
            CHECK(dtPtr);
            _ctx->_dataTableMap.push_back(dtPtr);
          } else {
            // TODO(croot): data table is changing. Need to validate any previously existing references.
            // One way to do this is store a map of all objects changing in-place in order to
            // validate.
            auto tableTypes = getDataTableTypesFromJSONObj(*vitr);
            if (dataTablePtr->getBaseType() != tableTypes.first || dataTablePtr->getType() != tableTypes.second) {
              // completely new data table type, so destroy previous one and
              // build a new one from scratch.
              auto& nameLookup = _ctx->_dataTableMap.get<QueryRendererContext::DataTableName>();
              auto itr = nameLookup.find(tableName);
              CHECK(itr != nameLookup.end());
              dataTablePtr = createDataTable(*vitr, dataObjPath, _ctx, tableName);
              auto dtPtr = std::dynamic_pointer_cast<BaseQueryDataTableJSON>(dataTablePtr);
              CHECK(dtPtr);
              nameLookup.replace(itr, dtPtr);

              dataEvents[static_cast<size_t>(RefEventType::REPLACE)].push_back(dtPtr);
            } else {
              auto dataTableJSONPtr = std::dynamic_pointer_cast<BaseQueryDataTableJSON>(dataTablePtr);
              CHECK(dataTableJSONPtr);
              if (dataTableJSONPtr->updateFromJSONObj(*vitr, dataObjPath)) {
                dataEvents[static_cast<size_t>(RefEventType::UPDATE)].push_back(dataTableJSONPtr);
              }
            }
          }

          unvisitedNames.erase(tableName);
          visitedNames.insert(std::move(tableName));
        }

        // now remove any unused tables that may be lingering around
        auto& nameLookup = _ctx->_dataTableMap.get<QueryRendererContext::DataTableName>();
        for (const auto& itr : unvisitedNames) {
          auto dataPtr = _ctx->getDataTable(itr);
          auto dataTableJSONPtr = std::dynamic_pointer_cast<BaseQueryDataTableJSON>(dataPtr);
          CHECK(dataTableJSONPtr);
          dataEvents[static_cast<size_t>(RefEventType::REMOVE)].push_back(dataTableJSONPtr);
          nameLookup.erase(itr);
        }
      } else {
        // need to clear out the previous data
        // TODO(croot): Need to invalidate any previous data references
        // This should probably be handled by some data reference object.
        // That or do an object validation check after everything's been rebuilt.
        // The latter would be the easiest way.
        _ctx->_dataTableMap.clear();
      }
    }

    std::array<std::vector<ScaleShPtr>, static_cast<size_t>(RefEventType::ALL)> scaleEvents;
    propName = "scales";
    {
      mitr = obj->FindMember(propName.c_str());
      if (mitr != obj->MemberEnd()) {
        rapidjson::Pointer scalePath = rootPath.Append(propName.c_str(), propName.length());

        RUNTIME_EX_ASSERT(
            mitr->value.IsArray(),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), mitr->value, "the \"" + propName + "\" member must be an array."));

        ScaleShPtr scalePtr;
        std::unordered_set<std::string> visitedNames;
        std::unordered_set<std::string> unvisitedNames;
        unvisitedNames.reserve(_ctx->_scaleConfigMap.size());
        for (auto& kv : _ctx->_scaleConfigMap) {
          unvisitedNames.insert(kv->getNameRef());
        }

        for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
          rapidjson::Pointer scaleObjPath = scalePath.Append(vitr - mitr->value.Begin());

          std::string scaleName = getScaleNameFromJSONObj(*vitr);

          RUNTIME_EX_ASSERT(
              visitedNames.find(scaleName) == visitedNames.end(),
              RapidJSONUtils::getJsonParseErrorStr(
                  _ctx->getUserWidgetIds(), *vitr, "a scale with the name \"" + scaleName + "\" already exists."));

          scalePtr = _ctx->getScale(scaleName);

          if (!scalePtr) {
            scalePtr = createScale(*vitr, scaleObjPath, _ctx, scaleName);
            _ctx->_scaleConfigMap.push_back(scalePtr);

            // TODO(croot): add an Add event type?
          } else {
            // TODO(croot): scale config is changing. Need to validate any previously existing references.
            // One way to do this is store a map of all objects changing in-place in order to
            // validate.
            auto currScaleType = getScaleTypeFromJSONObj(*vitr);
            QueryDataType rangeDataType;
            if (scalePtr->getType() != currScaleType ||
                scalePtr->getDomainDataType() != getScaleDomainDataTypeFromJSONObj(*vitr, _ctx, currScaleType) ||
                (rangeDataType = scalePtr->getRangeDataType()) !=
                    getScaleRangeDataTypeFromJSONObj(*vitr, _ctx, currScaleType) ||
                (rangeDataType == QueryDataType::COLOR &&
                 getScaleRangeColorTypeFromJSONObj(*vitr).first != scalePtr->getRangeColorType())) {
              // completely new scale type, so destroy previous one and
              // build a new one from scratch.

              auto& nameLookup = _ctx->_scaleConfigMap.get<QueryRendererContext::ScaleName>();
              auto itr = nameLookup.find(scaleName);
              CHECK(itr != nameLookup.end());
              // _ctx->_scaleConfigMap.erase(scaleName);
              scalePtr = createScale(*vitr, scaleObjPath, _ctx, scaleName);
              // _ctx->_scaleConfigMap.insert(std::make_pair(scaleName, scalePtr));
              nameLookup.replace(itr, scalePtr);

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
        auto& scaleNameLookup = _ctx->_scaleConfigMap.get<QueryRendererContext::ScaleName>();
        for (const auto& unvisitedName : unvisitedNames) {
          scalePtr = _ctx->getScale(unvisitedName);
          scaleNameLookup.erase(unvisitedName);
          _ctx->_accumulatorScales.erase(unvisitedName);
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
    }

    propName = "marks";
    {
      mitr = obj->FindMember(propName.c_str());
      if (mitr != obj->MemberEnd()) {
        rapidjson::Pointer markPath = rootPath.Append(propName.c_str(), propName.length());

        RUNTIME_EX_ASSERT(
            mitr->value.IsArray(),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), mitr->value, "the \"" + propName + "\" member must be an array"));

        size_t i;
        for (vitr = mitr->value.Begin(), i = 0; vitr != mitr->value.End(); ++vitr, ++i) {
          rapidjson::Pointer markObjPath = markPath.Append(vitr - mitr->value.Begin());

          if (i == _ctx->_geomConfigs.size()) {
            GeomConfigShPtr geomConfigPtr = createMark(*vitr, markObjPath, _ctx);

            _ctx->_geomConfigs.push_back(geomConfigPtr);
          } else {
            // do an update
            if (_ctx->_geomConfigs[i]->getType() != getMarkTypeFromJSONObj(*vitr)) {
              _ctx->_geomConfigs[i] = createMark(*vitr, markObjPath, _ctx);
            } else {
              _ctx->_geomConfigs[i]->updateFromJSONObj(*vitr, markObjPath);
            }
          }
        }
      } else {
        _ctx->_geomConfigs.clear();
      }
    }

    // now fire events so dependencies are cleaned and validated
    for (size_t i = 0; i < dataEvents.size(); ++i) {
      for (auto& dataPtr : dataEvents[i]) {
        _ctx->_fireRefEvent(static_cast<RefEventType>(i), dataPtr);
      }
    }

    for (size_t i = 0; i < scaleEvents.size(); ++i) {
      for (auto& scalePtr : scaleEvents[i]) {
        _ctx->_fireRefEvent(static_cast<RefEventType>(i), scalePtr);
      }
    }

    // validate the render order after all updates
    std::unordered_set<std::string> visitedAccumulators;
    std::string activeAccumulator, currAccumulator;

    for (size_t i = 0; i < _ctx->_geomConfigs.size(); ++i) {
      if (_ctx->_geomConfigs[i]->hasAccumulator()) {
        currAccumulator = _ctx->_geomConfigs[i]->getAccumulatorScaleName();
        if (activeAccumulator != currAccumulator) {
          activeAccumulator = currAccumulator;
          auto rtnPair = visitedAccumulators.insert(currAccumulator);
          RUNTIME_EX_ASSERT(rtnPair.second,
                            "Invalid render order. All geometry layers that use accumulator scales must be rendered "
                            "one after the other. There are at least 2 layers using the accumulator scale \"" +
                                currAccumulator + "\" that are separated by layers not using this accumulator.");
        }
      } else {
        activeAccumulator = "";
      }
    }

    _ctx->_jsonCache = jsonDocumentPtr;
  }

  // now update the data
  _ctx->_updateConfigGpuResources();

  // now update the pbo, if it is active, to ensure it is in alignment
  // with the used gpus after configuration.. this must be donw
  // after the gpu resources update above.
  _updatePbo();
}

size_t QueryRenderer::getWidth() {
  return _ctx->_width;
}

size_t QueryRenderer::getHeight() {
  return _ctx->_height;
}

void QueryRenderer::setWidthHeight(size_t width, size_t height) {
  RUNTIME_EX_ASSERT(width >= 0 && height >= 0,
                    to_string(_ctx->getUserWidgetIds()) + ": Cannot set a negative width/height.");

  _ctx->_width = width;
  _ctx->_height = height;

  _resizeFramebuffers(_ctx->_width, _ctx->_height);

  if (_ctx->doHitTest()) {
    // resize the cpu-bound pixels that store the ids per-pixel
    if (!_id1APixels) {
      _id1APixels.reset(new Array2dui(width, height));
    } else {
      _id1APixels->resize(width, height);
    }

    if (_ctx->supportsInt64()) {
      if (!_id1BPixels) {
        _id1BPixels.reset(new Array2dui(width, height));
      } else {
        _id1BPixels->resize(width, height);
      }
    }

    if (!_id2Pixels) {
      _id2Pixels.reset(new Array2di(width, height));
    } else {
      _id2Pixels->resize(width, height);
    }

    if (_pbo1A) {
      CHECK(_pbo2);
      auto qrmGpuCache = _ctx->getRootGpuCache();
      CHECK(qrmGpuCache);
      auto qrmPerGpuData = qrmGpuCache->perGpuData;
      CHECK(qrmPerGpuData);
      auto itr = qrmPerGpuData->find(_pboGpu);
      CHECK(itr != qrmPerGpuData->end());

      (*itr)->makeActiveOnCurrentThread();
      _pbo1A->resize(width, height);
      if (_pbo1B) {
        _pbo1B->resize(width, height);
      }
      _pbo2->resize(width, height);
    }
  }
}

void QueryRenderer::setJSONConfig(const std::string& configJSON, bool forceUpdate) {
  try {
    _initFromJSON(configJSON, forceUpdate);
  } catch (const ::Rendering::OutOfGpuMemoryError& e) {
    _clearAll();
    throw e;
  } catch (const ::Rendering::RenderError& e) {
    _clearAll(true);
    throw e;
  } catch (const std::exception& e) {
    _clearAll(true);
    throw std::runtime_error(e.what());
  }
}

void QueryRenderer::setJSONDocument(const std::shared_ptr<rapidjson::Document>& jsonDocumentPtr, bool forceUpdate) {
  try {
    _initFromJSON(jsonDocumentPtr, forceUpdate);
  } catch (const ::Rendering::OutOfGpuMemoryError& e) {
    _clearAll();
    throw e;
  } catch (const ::Rendering::RenderError& e) {
    _clearAll(true);
    throw e;
  } catch (const std::exception& e) {
    _clearAll(true);
    throw std::runtime_error(e.what());
  }
}

void QueryRenderer::updateResultsPostQuery(Executor* executor) {
  try {
    // now update the query result buffers
    _ctx->executor_ = executor;
  } catch (const ::Rendering::OutOfGpuMemoryError& e) {
    _clearAll();
    throw e;
  } catch (const ::Rendering::RenderError& e) {
    _clearAll(true);
    throw e;
  } catch (const std::exception& e) {
    _clearAll(true);
    throw std::runtime_error(e.what());
  }
}

void QueryRenderer::setQueryExecutionParams(Executor* executor,
                                            QueryExecCB execFunc,
                                            std::shared_ptr<RenderQueryExecuteTimer>& renderTimer) {
  _ctx->executor_ = executor;
  _ctx->execFunc_ = execFunc;
  _ctx->renderTimer_ = renderTimer;
}

void QueryRenderer::unsetQueryExecutionParams() {
  _ctx->executor_ = nullptr;
  _ctx->execFunc_ = nullptr;
  _ctx->renderTimer_ = nullptr;
}

void QueryRenderer::_update() {
  _ctx->_update();
}

void QueryRenderer::_createPbo(const std::set<GpuId>& usedGpus, int width, int height, bool makeContextInactive) {
  CHECK(!_pbo1A && !_pbo1B && !_pbo2);

  auto itr = usedGpus.begin();
  if (itr != usedGpus.end()) {
    auto qrmGpuCache = _ctx->getRootGpuCache();
    auto qrmPerGpuDataPtr = qrmGpuCache->perGpuData;
    CHECK(qrmPerGpuDataPtr);
    auto qrmItr = qrmPerGpuDataPtr->find(*itr);
    CHECK(qrmItr != qrmPerGpuDataPtr->end());

    if (usedGpus.size() > 1) {
      qrmItr = qrmPerGpuDataPtr->find((*qrmItr)->getCompositorGpuId());
      CHECK(qrmItr != qrmPerGpuDataPtr->end());
    }

    (*qrmItr)->makeActiveOnCurrentThread();

    auto widthToUse = (width < 0 ? _ctx->getWidth() : width);
    auto heightToUse = (height < 0 ? _ctx->getHeight() : height);
    _pbo1Awk = (*qrmItr)->getInactiveRowIdMapPbo(widthToUse, heightToUse);
    _pbo1A = _pbo1Awk.lock();
    if (_ctx->supportsInt64()) {
      _pbo1Bwk = (*qrmItr)->getInactiveRowIdMapPbo(widthToUse, heightToUse);
      _pbo1B = _pbo1Bwk.lock();
    }
    _pbo2wk = (*qrmItr)->getInactiveTableIdMapPbo(widthToUse, heightToUse);
    _pbo2 = _pbo2wk.lock();

    _pboGpu = (*qrmItr)->gpuId;

    if (makeContextInactive) {
      (*qrmItr)->makeInactive();
    }
  }
}

void QueryRenderer::_releasePbo(bool makeContextInactive) {
  if (_pbo1A || _pbo1B || _pbo2) {
    auto qrmGpuCache = _ctx->getRootGpuCache();
    if (qrmGpuCache) {
      auto qrmPerGpuData = _ctx->getRootGpuCache()->perGpuData;

      if (qrmPerGpuData) {
        auto itr = qrmPerGpuData->find(_pboGpu);
        CHECK(itr != qrmPerGpuData->end()) << "Couldn't find gpu data for gpuid: " << _pboGpu << ". Can't release pb";

        (*itr)->makeActiveOnCurrentThread();

        if (_pbo1A) {
          _pbo1A = nullptr;
          (*itr)->setRowIdMapPboInactive(_pbo1Awk);
        }

        if (_pbo1B) {
          _pbo1B = nullptr;
          (*itr)->setRowIdMapPboInactive(_pbo1Bwk);
        }

        if (_pbo2) {
          _pbo2 = nullptr;
          (*itr)->setTableIdMapPboInactive(_pbo2wk);
        }

        _pboGpu = EMPTY_GPUID;

        if (makeContextInactive) {
          (*itr)->makeInactive();
        }
      }
    }
  }
}

void QueryRenderer::_updatePbo() {
  if (_pbo1A || _pbo1B || _pbo2) {
    auto gpuIds = _ctx->getUsedGpus();

    bool deletePBO = (!gpuIds.size() || (gpuIds.size() == 1 && _pboGpu != *gpuIds.begin()));
    if (!deletePBO && gpuIds.size() > 1) {
      auto rootCache = _ctx->getRootGpuCache();
      CHECK(rootCache);
      auto itr = rootCache->perGpuData->find(_pboGpu);
      CHECK(itr != rootCache->perGpuData->end());
      if ((*itr)->getCompositorGpuId() != _pboGpu) {
        deletePBO = true;
      }
    }

    if (deletePBO) {
      _releasePbo();
    }
  }
}

void QueryRenderer::renderPasses(
    const std::shared_ptr<RootPerGpuDataMap>& qrmPerGpuData,
    QueryRendererContext* ctx,
    const std::set<GpuId>& usedGpus,
    bool clearFboEveryPass,
    std::function<
        void(::Rendering::GL::GLRenderer*, QueryFramebufferShPtr&, size_t, size_t, bool, bool, int, ScaleShPtr&, int)>
        perPassGpuCB,
    std::function<void(const std::set<GpuId>&, size_t, size_t, bool, bool, int, ScaleShPtr&)> passCompleteCB) {
  ScaleShPtr activeScale, prevScale, currScale;
  int accumulatorCnt = 0;
  int passCnt = 0;
  int gpuCnt = 0;
  auto width = ctx->getWidth();
  auto height = ctx->getHeight();
  auto doHitTest = ctx->doHitTest();
  auto doDepthTest = ctx->doDepthTest();

  for (size_t i = 0; i < ctx->_geomConfigs.size(); ++i) {
    currScale = ctx->_geomConfigs[i]->getAccumulatorScale();
    if (activeScale) {
      if (activeScale != currScale) {
        // finish the accumulation by running the blending pass
        // accumulationPassCompleteCB(scalePtr);
        passCompleteCB(usedGpus, width, height, doHitTest, doDepthTest, passCnt++, activeScale);
        prevScale = activeScale;

        if (!clearFboEveryPass && currScale) {
          // means we're doing an egl composite, or in other words
          // we're sharing framebuffers/textures across contexts
          // so we're not running clear all except on first pass
          // But that also means all textures need to be initialized
          // and marked as "shared" before any rendering. The
          // accumulationPreRender() call does this, so we need
          // to do that first.
          for (auto gpuId : usedGpus) {
            auto itr = qrmPerGpuData->find(gpuId);
            (*itr)->makeActiveOnCurrentThread();
            currScale->accumulationPreRender(gpuId);
          }
        }
        accumulatorCnt = 0;
      } else {
        accumulatorCnt++;
      }
    } else if (currScale) {
      prevScale = nullptr;

      if (!clearFboEveryPass) {
        for (auto gpuId : usedGpus) {
          auto itr = qrmPerGpuData->find(gpuId);
          (*itr)->makeActiveOnCurrentThread();
          currScale->accumulationPreRender(gpuId);
        }
      }
    } else {
      prevScale = nullptr;
      accumulatorCnt = 0;
    }
    activeScale = currScale;

    gpuCnt = 0;
    for (auto gpuId : usedGpus) {
      auto itr = qrmPerGpuData->find(gpuId);
      CHECK(itr != qrmPerGpuData->end());
      (*itr)->makeActiveOnCurrentThread();

      if (prevScale) {
        prevScale->accumulationPostRender(gpuId);
      }

      if (clearFboEveryPass && accumulatorCnt == 0 && activeScale) {
        activeScale->accumulationPreRender(gpuId);
      }

      auto renderer = (*itr)->getGLRenderer();
      CHECK(renderer != nullptr);

      auto& framebufferPtr = (*itr)->getRenderFramebuffer();

      RUNTIME_EX_ASSERT(framebufferPtr != nullptr,
                        "QueryRenderer " + to_string(ctx->getUserWidgetIds()) +
                            ": The framebuffer is not initialized for gpu " + std::to_string(gpuId) +
                            ". Cannot render.");

      // need to set the hit test / depth test before the bindToRenderer()
      // TODO(croot): may want to change this API

      if (i == 0) {
        framebufferPtr->setHitTest(ctx->doHitTest());
        framebufferPtr->setDepthTest(ctx->doDepthTest());

        // TODO(croot): enable a push/pop state stack for the renderer state...
        renderer->enable(GL_BLEND);
        renderer->setBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        renderer->setBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

        renderer->setViewport(0, 0, ctx->_width, ctx->_height);
        renderer->setClearColor(0, 0, 0, 0);
      }

      framebufferPtr->bindToRenderer(renderer);

      if (clearFboEveryPass || (i == 0 && gpuCnt == 0)) {
        renderer->clearAll();
      }

      // TODO(croot): only draw geom on this gpu if the geom is configured to use
      // this gpu
      ctx->_geomConfigs[i]->draw(renderer, gpuId);

      perPassGpuCB(
          renderer, framebufferPtr, width, height, doHitTest, doDepthTest, gpuCnt++, activeScale, accumulatorCnt);
    }

    if (!activeScale) {
      passCompleteCB(usedGpus, width, height, doHitTest, doDepthTest, passCnt++, activeScale);
    }

    // NOTE: We're not swapping buffers because we're using pbuffers

    // CROOT testing code
    // int width = ctx->getWidth();
    // int height = ctx->getHeight();
    // std::shared_ptr<unsigned char> pixelsPtr;
    // pixelsPtr = framebufferPtr->readColorBuffer(0, 0, width, height);

    // PngData pngData(width, height, pixelsPtr);
    // pngData.writeToFile("render_" + std::to_string(gpuId) + ".png");
  }

  if (activeScale) {
    // finish the accumulation by running the blending pass
    // accumulationPassCompleteCB(scalePtr)
    passCompleteCB(usedGpus, width, height, doHitTest, doDepthTest, passCnt++, activeScale);

    for (auto gpuId : usedGpus) {
      auto itr = qrmPerGpuData->find(gpuId);
      (*itr)->makeActiveOnCurrentThread();
      activeScale->accumulationPostRender(gpuId);
    }
  }
}

QueryFramebufferShPtr& QueryRenderer::renderGpu(GpuId gpuId,
                                                const std::shared_ptr<RootPerGpuDataMap>& qrmPerGpuData,
                                                QueryRendererContext* ctx,
                                                int r,
                                                int g,
                                                int b,
                                                int a) {
  // TODO(croot): make thread safe?

  ScaleShPtr activeScale, currScale;
  auto itr = qrmPerGpuData->find(gpuId);

  CHECK(itr != qrmPerGpuData->end());

  (*itr)->makeActiveOnCurrentThread();

  GLRenderer* renderer = (*itr)->getGLRenderer();
  CHECK(renderer != nullptr);

  auto& framebufferPtr = (*itr)->getRenderFramebuffer();

  RUNTIME_EX_ASSERT(framebufferPtr != nullptr,
                    "QueryRenderer " + to_string(ctx->getUserWidgetIds()) +
                        ": The framebuffer is not initialized for gpu " + std::to_string(gpuId) + ". Cannot render.");

  // need to set the hit test / depth test before the bindToRenderer()
  // TODO(croot): may want to change this API
  framebufferPtr->setHitTest(ctx->doHitTest());
  framebufferPtr->setDepthTest(ctx->doDepthTest());
  framebufferPtr->bindToRenderer(renderer);

  // TODO(croot): enable a push/pop state stack for the renderer state...
  renderer->enable(GL_BLEND);
  renderer->setBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
  renderer->setBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

  renderer->setViewport(0, 0, ctx->_width, ctx->_height);
  if (r >= 0 && g >= 0 && b >= 0 && a >= 0) {
    renderer->setClearColor(r, g, b, a);
    renderer->clearAll();
  }

  for (size_t i = 0; i < ctx->_geomConfigs.size(); ++i) {
    currScale = ctx->_geomConfigs[i]->getAccumulatorScale();
    if (activeScale != currScale) {
      if (activeScale) {
        // finish the accumulation by running the blending pass
        activeScale->renderAccumulation(renderer, gpuId);
        activeScale->accumulationPostRender(gpuId);
      }

      if (currScale) {
        currScale->accumulationPreRender(gpuId);
      }
    }
    activeScale = currScale;

    // TODO(croot): only draw geom on this gpu if the geom is configured to use
    // this gpu
    ctx->_geomConfigs[i]->draw(renderer, gpuId);
  }

  if (activeScale) {
    // finish the accumulation by running the blending pass
    activeScale->renderAccumulation(renderer, gpuId);
    activeScale->accumulationPostRender(gpuId);
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

  return framebufferPtr;
}

void QueryRenderer::render(bool inactivateRendererOnThread) {
  _render(_ctx->getUsedGpus(), inactivateRendererOnThread);
}

void QueryRenderer::_render(const std::set<GpuId>& usedGpus, bool inactivateRendererOnThread) {
  try {
    // update everything marked dirty before rendering
    _update();

    if (_ctx->doHitTest() && ((!_pbo1A || !_pbo2) || (_ctx->supportsInt64() && !_pbo1B))) {
      _createPbo(usedGpus);
    }

    int numGpusToRender = usedGpus.size();
    if (numGpusToRender) {
      auto qrmGpuCache = _ctx->getRootGpuCache();
      CHECK(qrmGpuCache);
      auto qrmPerGpuDataPtr = qrmGpuCache->perGpuData;
      auto qrmItr = qrmPerGpuDataPtr->find((*usedGpus.begin()));
      CHECK(qrmItr != qrmPerGpuDataPtr->end());
      if (numGpusToRender == 1) {
        auto usedFramebuffer = renderGpu((*qrmItr)->gpuId, qrmPerGpuDataPtr, _ctx.get(), 0, 0, 0, 0).get();
        auto aaFramebuffer = _runAntialiasingPass(*qrmItr, usedFramebuffer);

        if (aaFramebuffer->getNumSamples() > 1) {
          // need to blit the multisampled fbo into a non-sampled fbo
          auto& blitFramebuffer = (*qrmItr)->getAntiAliasingFramebuffer();
          CHECK(blitFramebuffer && blitFramebuffer->getGLRenderer() == usedFramebuffer->getGLRenderer());
          usedFramebuffer->blitToFramebuffer(*blitFramebuffer, 0, 0, _ctx->getWidth(), _ctx->getHeight());
          usedFramebuffer = blitFramebuffer.get();
        }

        if (_ctx->doHitTest()) {
          CHECK(_id1APixels && _id2Pixels && _pbo1A && _pbo2 &&
                usedFramebuffer->getGLRenderer()->getGpuId() == _pboGpu);
          usedFramebuffer->copyRowIdBufferToPbo(_pbo1A, true);
          if (_id1BPixels && _pbo1B) {
            usedFramebuffer->copyRowIdBufferToPbo(_pbo1B, false);
          }
          usedFramebuffer->copyTableIdBufferToPbo(_pbo2);
        }
      } else {
        auto compId = (*qrmItr)->getCompositorGpuId();
        if (compId != (*qrmItr)->gpuId) {
          qrmItr = qrmPerGpuDataPtr->find(compId);
          CHECK(qrmItr != qrmPerGpuDataPtr->end());
        }

        auto& compositorPtr = (*qrmItr)->getCompositor();
        CHECK(compositorPtr);

        auto usedFramebuffer = compositorPtr->render(this, usedGpus).get();

        if (usedFramebuffer->getNumSamples() > 1) {
          // need to blit the multisampled fbo into a non-sampled fbo
          auto& blitFramebuffer = (*qrmItr)->getAntiAliasingFramebuffer();
          CHECK(blitFramebuffer && blitFramebuffer->getGLRenderer() == usedFramebuffer->getGLRenderer());
          auto renderer = usedFramebuffer->getRenderer();
          renderer->makeActiveOnCurrentThread();
          usedFramebuffer->blitToFramebuffer(*blitFramebuffer, 0, 0, _ctx->getWidth(), _ctx->getHeight());
          usedFramebuffer = blitFramebuffer.get();
        }

        auto aaFramebuffer = _runAntialiasingPass(*qrmItr, usedFramebuffer);

        if (_ctx->doHitTest()) {
          CHECK(_id1APixels && _id2Pixels && _pbo1A && _pbo2 &&
                usedFramebuffer->getGLRenderer()->getGpuId() == _pboGpu);
          auto renderer = usedFramebuffer->getRenderer();
          renderer->makeActiveOnCurrentThread();

          usedFramebuffer->copyRowIdBufferToPbo(_pbo1A, true);
          if (_id1BPixels && _pbo1B) {
            usedFramebuffer->copyRowIdBufferToPbo(_pbo1B, false);
          }
          usedFramebuffer->copyTableIdBufferToPbo(_pbo2);
        }
      }

      _idPixelsDirty = true;
    }

    if (inactivateRendererOnThread) {
      GLRenderer* currRenderer = GLRenderer::getCurrentThreadRenderer();
      if (currRenderer) {
        currRenderer->makeInactive();
      }
    }
  } catch (const ::Rendering::OutOfGpuMemoryError& e) {
    _clearAll();
    throw e;
  } catch (const ::Rendering::RenderError& e) {
    _clearAll(true);
    throw e;
  } catch (const std::exception& e) {
    _clearAll(true);
    throw std::runtime_error(e.what());
  }
}

QueryFramebufferShPtr& QueryRenderer::_runAntialiasingPass(const RootPerGpuDataShPtr& gpuData,
                                                           QueryFramebuffer* rndrFbo) {
  auto aaPassPtr = gpuData->getSMAAPassPtr();
  auto renderer = rndrFbo->getGLRenderer();
  auto& outputFbo = gpuData->getAntiAliasingFramebuffer();
  CHECK(aaPassPtr && renderer->getGpuId() == gpuData->gpuId && outputFbo);

  auto inFbo = rndrFbo->getGLFramebuffer();
  auto outFbo = outputFbo->getGLFramebuffer();
  aaPassPtr->runPass(_ctx->_width, _ctx->_height, renderer, inFbo, outFbo);

  return outputFbo;
}

QueryRenderer::RenderedPixels QueryRenderer::_renderAndGetPixels() {
  auto usedGpus = _ctx->getUsedGpus();
  _render(usedGpus, false);
  bool empty = false;

  try {
    int width = getWidth();
    int height = getHeight();
    std::shared_ptr<unsigned char> pixelsPtr = nullptr;

    int numGpus = usedGpus.size();
    if (numGpus) {
      auto qrmGpuCache = _ctx->getRootGpuCache();
      CHECK(qrmGpuCache);
      auto qrmPerGpuDataPtr = qrmGpuCache->perGpuData;
      auto itr = qrmPerGpuDataPtr->find(*usedGpus.begin());
      CHECK(itr != qrmPerGpuDataPtr->end());

      if (numGpus > 1) {
        auto compId = (*itr)->getCompositorGpuId();
        if (compId != (*itr)->gpuId) {
          itr = qrmPerGpuDataPtr->find(compId);
          CHECK(itr != qrmPerGpuDataPtr->end());
        }

        auto& compositorPtr = (*itr)->getCompositor();
        CHECK(compositorPtr);

        auto renderer = compositorPtr->getGLRenderer();
        renderer->makeActiveOnCurrentThread();

        auto framebufferPtr = (*itr)->getAntiAliasingFramebuffer().get();
        if (framebufferPtr) {
          framebufferPtr->bindToRenderer(renderer, FboBind::READ);
          pixelsPtr = framebufferPtr->readColorBuffer(0, 0, width, height);
        } else {
          pixelsPtr = compositorPtr->readColorBuffer(0, 0, width, height);
        }
        renderer->makeInactive();
      } else {
        auto renderer = (*itr)->getGLRenderer();
        auto framebufferPtr = (*itr)->getAntiAliasingFramebuffer().get();
        if (!framebufferPtr) {
          framebufferPtr = (*itr)->getRenderFramebuffer().get();
        }
        CHECK(renderer != nullptr);
        CHECK(framebufferPtr && framebufferPtr->getGLRenderer() == renderer);

        renderer->makeActiveOnCurrentThread();

        framebufferPtr->bindToRenderer(renderer, FboBind::READ);
        pixelsPtr = framebufferPtr->readColorBuffer(0, 0, width, height);

        renderer->makeInactive();
      }
    } else {
      empty = true;
      if (width > 0 && height > 0) {
        pixelsPtr.reset(new unsigned char[width * height * 4], std::default_delete<unsigned char[]>());
        std::memset(pixelsPtr.get(), 0, width * height * 4 * sizeof(unsigned char));

        if (_ctx->doHitTest()) {
          // clear out the id buffer
          _id1APixels->resetToDefault();
          if (_id1BPixels) {
            _id1BPixels->resetToDefault();
          }
          _id2Pixels->resetToDefault();
        }
      }
    }

    return RenderedPixels(pixelsPtr, empty);
  } catch (const ::Rendering::OutOfGpuMemoryError& e) {
    _clearAll();
    throw e;
  } catch (const ::Rendering::RenderError& e) {
    _clearAll(true);
    throw e;
  } catch (const std::exception& e) {
    _clearAll(true);
    throw std::runtime_error(e.what());
  }
}

RawPixelData QueryRenderer::renderRawData() {
  auto renderedPixels = _renderAndGetPixels();
  // make sure our cached ids are updated post-render
  _updateIdBuffers();
  if (renderedPixels.pixels && !renderedPixels.isEmptyRender) {
    return RawPixelData(getWidth(),
                        getHeight(),
                        4,
                        renderedPixels.pixels,
                        (_id1APixels ? _id1APixels->getDataShPtr() : nullptr),
                        (_id1BPixels ? _id1BPixels->getDataShPtr() : nullptr),
                        (_id2Pixels ? _id2Pixels->getDataShPtr() : nullptr));
  } else {
    return RawPixelData(getWidth(), getHeight());
  }
}

PngData QueryRenderer::renderToPng(int compressionLevel) {
  RUNTIME_EX_ASSERT(compressionLevel >= -1 && compressionLevel <= 9,
                    "QueryRenderer " + to_string(_ctx->getUserWidgetIds()) + ": Invalid compression level " +
                        std::to_string(compressionLevel) + ". It must be a " +
                        "value between 0 (no zlib compression) to 9 (most zlib compression), or -1 (use default).");

  auto renderedPixels = _renderAndGetPixels();

  try {
    return PngData(getWidth(), getHeight(), renderedPixels.pixels, compressionLevel);
  } catch (const ::Rendering::OutOfGpuMemoryError& e) {
    _clearAll();
    throw e;
  } catch (const ::Rendering::RenderError& e) {
    _clearAll(true);
    throw e;
  } catch (const std::exception& e) {
    _clearAll(true);
    throw std::runtime_error(e.what());
  }
}

PngData QueryRenderer::compositeRenderBuffersToPng(const std::vector<RawPixelData>& buffers, int compressionLevel) {
  RUNTIME_EX_ASSERT(compressionLevel >= -1 && compressionLevel <= 9,
                    "QueryRenderer " + to_string(_ctx->getUserWidgetIds()) + ": Invalid compression level " +
                        std::to_string(compressionLevel) + ". It must be a " +
                        "value between 0 (no zlib compression) to 9 (most zlib compression), or -1 (use default).");

  try {
    // update everything marked dirty before rendering
    _update();

    std::shared_ptr<unsigned char> pixelsPtr = nullptr;
    auto gpuCache = _ctx->getRootGpuCache();
    auto distCompositorPtr = gpuCache->getDistributedCompositorPtr();
    auto renderer = distCompositorPtr->getGLRenderer();
    CHECK(renderer);

    if (_ctx->doHitTest() && ((!_pbo1A || !_pbo2) || (_ctx->supportsInt64() && !_pbo1B))) {
      _createPbo({renderer->getGpuId()});
    }

    size_t width = getWidth(), height = getHeight();
    if (buffers.size()) {
      width = buffers[0].width;
      height = buffers[0].height;
    }

    if (width > 0 && height > 0) {
      setWidthHeight(width, height);
      if (buffers.size()) {
        renderer->makeActiveOnCurrentThread();
        distCompositorPtr->render(buffers);

        auto fbo = distCompositorPtr->getFramebufferPtr();
        CHECK(fbo);
        if (_ctx->doHitTest()) {
          // copy row/table id data over to the pbos for lazy device->host transfer
          CHECK(_id1APixels && _id2Pixels && _pbo1A && _pbo2 && renderer->getGpuId() == _pboGpu);
          fbo->copyRowIdBufferToPbo(_pbo1A, true);
          if (_id1BPixels && _pbo1B) {
            fbo->copyRowIdBufferToPbo(_pbo1B, false);
          }
          fbo->copyTableIdBufferToPbo(_pbo2);
        }

        // now read the color buffer from the render to png encode
        fbo->bindToRenderer(distCompositorPtr->getGLRenderer(), FboBind::READ);
        pixelsPtr = fbo->readColorBuffer(0, 0, width, height);

        // mark the host row id/table id caches as dirty.
        _idPixelsDirty = true;
      } else {
        pixelsPtr.reset(new unsigned char[width * height * 4], std::default_delete<unsigned char[]>());
        std::memset(pixelsPtr.get(), 0, width * height * 4 * sizeof(unsigned char));
        if (_ctx->doHitTest()) {
          // clear out the id buffer
          _id1APixels->resetToDefault();
          if (_id1BPixels) {
            _id1BPixels->resetToDefault();
          }
          _id2Pixels->resetToDefault();
        }
      }
    }

    return PngData(width, height, pixelsPtr, compressionLevel);
  } catch (const ::Rendering::OutOfGpuMemoryError& e) {
    _clearAll();
    throw e;
  } catch (const ::Rendering::RenderError& e) {
    _clearAll(true);
    throw e;
  } catch (const std::exception& e) {
    _clearAll(true);
    throw std::runtime_error(e.what());
  }
}

bool QueryRenderer::_updateIdBuffers() {
  if (_id1APixels && _id2Pixels) {
    if (_idPixelsDirty) {
      size_t width = _ctx->getWidth();
      size_t height = _ctx->getHeight();
      CHECK(_id1APixels->getWidth() == width && _id1APixels->getHeight() == height && _id2Pixels->getWidth() == width &&
            _id2Pixels->getHeight() == height);

      unsigned int* rawIds1A = _id1APixels->getDataPtr();
      int* rawIds2 = _id2Pixels->getDataPtr();

      auto qrmGpuCache = _ctx->getRootGpuCache();
      auto qrmPerGpuData = qrmGpuCache->perGpuData;
      CHECK(qrmPerGpuData);
      auto itr = qrmPerGpuData->find(_pboGpu);

      if (itr != qrmPerGpuData->end()) {
        (*itr)->makeActiveOnCurrentThread();
        _pbo1A->readIdBuffer(width, height, rawIds1A);
        if (_id1BPixels && _pbo1B) {
          auto rawIds1B = _id1BPixels->getDataPtr();
          _pbo1B->readIdBuffer(width, height, rawIds1B);
        }
        _pbo2->readIdBuffer(width, height, rawIds2);
        _releasePbo();
        (*itr)->makeInactive();
      }

      _idPixelsDirty = false;
    }
    return true;
  }
  return false;
}

HitInfo QueryRenderer::getIdAt(size_t x, size_t y, size_t pixelRadius) {
  RUNTIME_EX_ASSERT(_ctx->doHitTest(),
                    "QueryRenderer " + to_string(_ctx->getUserWidgetIds()) + " was not initialized for hit-testing.");

  try {
    uint64_t id = 0;
    int tableId = 0;

    if (_id1APixels && _id2Pixels) {
      size_t width = _ctx->getWidth();
      size_t height = _ctx->getHeight();
      if (x < width && y < height) {
        // make sure we have fully updated our cached ids first before accessing
        _updateIdBuffers();

        if (pixelRadius == 0) {
          id = static_cast<uint64_t>(_id1APixels->get(x, y));
          if (_id1BPixels) {
            id |= (static_cast<uint64_t>(_id1BPixels->get(x, y)) << 32);
          }
          tableId = _id2Pixels->get(x, y);
        } else {
          typedef std::unordered_map<size_t, Array2df> KernelMap;
          static KernelMap gaussKernels;

          size_t pixelRadius2xPlus1 = pixelRadius * 2 + 1;
          auto itr = gaussKernels.find(pixelRadius);
          if (itr == gaussKernels.end()) {
            itr = gaussKernels.insert(itr, std::make_pair(pixelRadius, createGaussianKernel(pixelRadius2xPlus1, 0.75)));
          }

          const Array2df& kernel = itr->second;
          Array2dui ids1A(pixelRadius2xPlus1, pixelRadius2xPlus1);
          Array2dui ids1B(pixelRadius2xPlus1, pixelRadius2xPlus1);
          Array2di ids2(pixelRadius2xPlus1, pixelRadius2xPlus1);

          ids1A.copyFromPixelCenter(*_id1APixels,
                                    x,
                                    y,
                                    pixelRadius,
                                    pixelRadius,
                                    pixelRadius,
                                    pixelRadius,
                                    ::Rendering::Objects::WrapType::USE_DEFAULT);

          if (_id1BPixels) {
            ids1B.copyFromPixelCenter(*_id1BPixels,
                                      x,
                                      y,
                                      pixelRadius,
                                      pixelRadius,
                                      pixelRadius,
                                      pixelRadius,
                                      ::Rendering::Objects::WrapType::USE_DEFAULT);
          }

          ids2.copyFromPixelCenter(*_id2Pixels,
                                   x,
                                   y,
                                   pixelRadius,
                                   pixelRadius,
                                   pixelRadius,
                                   pixelRadius,
                                   ::Rendering::Objects::WrapType::USE_DEFAULT);

          // build up the counter
          std::unordered_map<unsigned int, std::unordered_map<uint64_t, float>> idCounterMap;
          decltype(idCounterMap)::iterator tableIdItr;
          std::unordered_map<uint64_t, float>::iterator idItr;
          for (size_t i = 0; i < pixelRadius2xPlus1; ++i) {
            for (size_t j = 0; j < pixelRadius2xPlus1; ++j) {
              id = static_cast<uint64_t>(ids1A[i][j]);
              id |= (static_cast<uint64_t>(ids1B[i][j]) << 32);
              tableId = ids2[i][j];

              // TODO(croot): what about cases where there's no table id defined?
              if (id > 0 && kernel[i][j] > 0) {  // don't include empty pixels or gaussian distro outliers
                if ((tableIdItr = idCounterMap.find(tableId)) == idCounterMap.end()) {
                  tableIdItr = idCounterMap.insert(
                      tableIdItr, std::make_pair(tableId, std::unordered_map<uint64_t, float>({{id, 0.0}})));
                }

                if ((idItr = tableIdItr->second.find(id)) == tableIdItr->second.end()) {
                  idItr = tableIdItr->second.insert(idItr, std::make_pair(id, 0.0));
                }
                idItr->second += kernel[i][j];
              }
            }
          }

          if (idCounterMap.size() == 0) {
            id = 0;
            tableId = 0;
          } else if (idCounterMap.size() == 1) {
            tableIdItr = idCounterMap.begin();
            if (tableIdItr->second.size() == 1) {
              id = tableIdItr->second.begin()->first;
            } else {
              idItr = std::max_element(tableIdItr->second.begin(), tableIdItr->second.end(), idCounterCompare);
              id = idItr->first;
            }
            tableId = tableIdItr->first;
          } else {
            std::vector<std::tuple<unsigned int, uint64_t, float>> flattened(idCounterMap.size());

            size_t idx = 0;
            std::for_each(idCounterMap.begin(), idCounterMap.end(), [&flattened, &idx](auto& item) {
              auto itr = std::max_element(item.second.begin(), item.second.end(), idCounterCompare);
              flattened[idx++] = std::make_tuple(item.first, itr->first, itr->second);
            });

            auto itr = std::max_element(flattened.begin(), flattened.end(), [](auto& item, auto& item2) {
              return std::get<2>(item) < std::get<2>(item2);
            });
            id = std::get<1>(*itr);
            tableId = std::get<0>(*itr);
          }
        }
      }
    }

    auto rtnTableId = static_cast<TableId>((std::abs(tableId) >> 5) * (std::signbit(tableId) ? -1 : 1));
    auto dataId = static_cast<uint8_t>(std::abs(tableId) & 31);
    return HitInfo(rtnTableId, id, dataId);
  } catch (const ::Rendering::OutOfGpuMemoryError& e) {
    _clearAll();
    throw e;
  } catch (const ::Rendering::RenderError& e) {
    _clearAll(true);
    throw e;
  } catch (const std::exception& e) {
    _clearAll(true);
    throw std::runtime_error(e.what());
  }
}

std::string QueryRenderer::getVegaTableNameFromIndex(const int8_t dataId) const {
  if (dataId < 0 || static_cast<size_t>(dataId) >= _ctx->_dataTableMap.size()) {
    return "";
  }

  return _ctx->_dataTableMap[dataId]->getName();
}

QueryDataTableBaseType QueryRenderer::getVegaTableTypeWithTableId(const TableId tableId) const {
  RUNTIME_EX_ASSERT(tableId != NonProjectionRenderQueryCacheMap::emptyTableId,
                    "TableId " + std::to_string(tableId) + " is not a valid table id.");

  // TODO(croot): improve the performance here with a different data structure
  // for the data objects that is query-able by table id?
  // I don't expect the number of tables in a vega to be very many, so just iterating
  // over them all shouldn't be a big deal for now.
  QueryDataTableBaseType baseType = QueryDataTableBaseType::UNSUPPORTED;
  std::for_each(_ctx->_dataTableMap.begin(), _ctx->_dataTableMap.end(), [&](const auto& dataTablePtr) {
    auto sqlTablePtr = std::dynamic_pointer_cast<BaseQueryDataTableSQL>(dataTablePtr);
    if (sqlTablePtr && sqlTablePtr->getTableId() == tableId) {
      auto baseTablePtr = std::dynamic_pointer_cast<BaseQueryDataTable>(dataTablePtr);
      CHECK(baseTablePtr);
      if (baseType == QueryDataTableBaseType::UNSUPPORTED) {
        baseType = baseTablePtr->getBaseType();
      } else {
        RUNTIME_EX_ASSERT(baseTablePtr->getBaseType() == baseType,
                          to_string(_ctx->getUserWidgetIds()) +
                              ": There is more than 1 table in the vega with the same table id, but the tables have "
                              "differing table types. Cannot distinguish the table type for table id " +
                              std::to_string(tableId));
      }
    }
  });

  RUNTIME_EX_ASSERT(baseType != QueryDataTableBaseType::UNSUPPORTED,
                    to_string(_ctx->getUserWidgetIds()) + ": The table id " + std::to_string(tableId) +
                        " is not used by any of the vega tables.");

  return baseType;
}

}  // namespace QueryRenderer
