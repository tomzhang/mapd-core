#include "BaseMark.h"
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include "../Scales/Utils.h"
#include "../Utils/ShaderUtils.h"

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;
using ::Rendering::GL::Resources::VboAttrToShaderAttrMap;

BaseMark::BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx)
    : _type(geomType),
      key(this, "key", ctx, false, false),
      _invalidKey(std::numeric_limits<int64_t>::max()),
      _dataPtr(nullptr),
      _perGpuData(),
      _ctx(ctx),
      _shaderDirty(true),
      _propsDirty(true),
      _vboProps(),
      _uboProps(),
      _uniformProps(),
      _activeAccumulatorScaleName("") {
}

BaseMark::BaseMark(GeomType geomType,
                   const QueryRendererContextShPtr& ctx,
                   const rapidjson::Value& obj,
                   const rapidjson::Pointer& objPath,
                   QueryDataTableBaseType baseType,
                   bool mustUseDataRef)
    : BaseMark(geomType, ctx) {
  _initFromJSONObj(obj, objPath, baseType, mustUseDataRef, true);
}

BaseMark::~BaseMark() {
}

void BaseMark::setAccumulatorScale(const std::string& accumulatorScaleName) {
  RUNTIME_EX_ASSERT(
      !_activeAccumulatorScaleName.size() || _activeAccumulatorScaleName == accumulatorScaleName,
      std::string(*this) + ": An accumulator scale named: \"" + _activeAccumulatorScaleName +
          "\" is already set active on the mark. Only one accumulator scale per mark is currently supported.");

  _activeAccumulatorScaleName = accumulatorScaleName;
}

void BaseMark::_initFromJSONObj(const rapidjson::Value& obj,
                                const rapidjson::Pointer& objPath,
                                QueryDataTableBaseType baseType,
                                bool mustUseDataRef,
                                bool initializing) {
  RUNTIME_EX_ASSERT(
      obj.IsObject(),
      RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(), obj, "definition for marks must be an object."));

  rapidjson::Value::ConstMemberIterator mitr;

  // TODO(croot): move the "from" to a const somewhere
  std::string fromProp = "from";
  if ((mitr = obj.FindMember(fromProp.c_str())) != obj.MemberEnd()) {
    const rapidjson::Value& fromObj = mitr->value;

    if (!_ctx->isJSONCacheUpToDate(_dataPtrJsonPath, fromObj)) {
      RUNTIME_EX_ASSERT(fromObj.IsObject(),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(), fromObj, "mark data reference must be an object."));

      RUNTIME_EX_ASSERT(
          (mitr = fromObj.FindMember("data")) != fromObj.MemberEnd() && mitr->value.IsString(),
          RapidJSONUtils::getJsonParseErrorStr(
              _ctx->getUserWidgetIds(), fromObj, "mark data reference must contain a \"data\" string property."));

      _dataPtr = _ctx->getDataTable(mitr->value.GetString());

      RUNTIME_EX_ASSERT(_dataPtr->getBaseType() == baseType,
                        RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                                             fromObj,
                                                             "mark data table reference is the wrong type. It is a " +
                                                                 to_string(_dataPtr->getBaseType()) + " table but a " +
                                                                 to_string(baseType) + " table is required."));
    }

    _dataPtrJsonPath = objPath.Append(fromProp.c_str(), fromProp.length());
  } else {
    // TODO(croot): what about references???
    _dataPtr = nullptr;
    RUNTIME_EX_ASSERT(!mustUseDataRef,
                      RapidJSONUtils::getJsonParseErrorStr(
                          _ctx->getUserWidgetIds(),
                          obj,
                          "A data reference (i.e. \"" + fromProp + "\") is not defined for mark. It is required."));
  }

  _initGpuResources(_ctx.get(), initializing);
}

void BaseMark::_updateProps(const std::set<BaseRenderProperty*>& usedProps, bool force) {
  // Now update which props are vbo-defined, and which will be uniforms
  size_t sum = _vboProps.size() + _uboProps.size() + _uniformProps.size();
  CHECK(sum == 0 || usedProps.size() == sum);

  if (force) {
    _vboProps.clear();
    _uboProps.clear();
    _uniformProps.clear();

    for (const auto& prop : usedProps) {
      if (prop->hasVboPtr()) {
        _vboProps.insert(prop);
      } else if (prop->hasUboPtr()) {
        _uboProps.insert(prop);
      } else {
        _uniformProps.insert(prop);
      }
    }

    setShaderDirty();
  } else {
    for (const auto& prop : usedProps) {
      if (prop->hasVboPtr()) {
        if (_vboProps.insert(prop).second) {
          setShaderDirty();
          if (!_uboProps.erase(prop)) {
            _uniformProps.erase(prop);
          }
        }
      } else if (prop->hasUboPtr()) {
        if (_uboProps.insert(prop).second) {
          setShaderDirty();
          if (!_vboProps.erase(prop)) {
            _uniformProps.erase(prop);
          }
        }
      } else {
        if (_uniformProps.insert(prop).second) {
          setShaderDirty();
          if (!_vboProps.erase(prop)) {
            _uboProps.erase(prop);
          }
        }
      }
    }
  }
}

void BaseMark::_updateShader(std::string& vertSrc, std::string& fragSrc) {
  if (hasAccumulator()) {
    const std::string colorFuncName = "getFragColor";
    auto funcRange = ShaderUtils::getGLSLFunctionBounds(fragSrc, colorFuncName);

    RUNTIME_EX_ASSERT(!funcRange.empty(),
                      std::string(*this) + ": Cannot find a properly defined \"" + colorFuncName +
                          "\" function in the fragment shader.");

    auto scalePtr = _ctx->getScale(getAccumulatorScaleName());

    std::string accumSrc = scalePtr->getAccumulator1stPassFragSrc();

    boost::replace_range(fragSrc, funcRange, accumSrc);
  }
}

void BaseMark::_buildVertexArrayObjectFromProperties() {
  if (!_propsDirty || !_perGpuData.size()) {
    // early out
    return;
  }

  _updateProps(_getUsedProps());

  // TODO(croot): make thread safe?
  // GLRenderer* prevRenderer = GLRenderer::getCurrentThreadRenderer();
  // Window* prevWindow = GLRenderer::getCurrentThreadWindow();
  GLRenderer* currRenderer = nullptr;

  RootPerGpuDataShPtr qrmGpuData;
  for (auto& itr : _perGpuData) {
    qrmGpuData = itr.second.getRootPerGpuData();
    CHECK(qrmGpuData);

    qrmGpuData->makeActiveOnCurrentThread();

    currRenderer = dynamic_cast<GLRenderer*>(qrmGpuData->rendererPtr.get());

    CHECK(currRenderer != nullptr);

    GLResourceManagerShPtr rsrcMgr = currRenderer->getResourceManager();

    if (itr.second.shaderPtr) {
      currRenderer->bindShader(itr.second.shaderPtr);

      // build property map for how vertex buffer attributes will
      // be bound to shader attributes
      VboAttrToShaderAttrMap attrMap;
      ::Rendering::GL::Resources::GLIndexBufferShPtr ibo;
      _buildVAOData(itr.first, itr.second.shaderPtr.get(), attrMap, ibo);

      itr.second.vaoPtr = rsrcMgr->createVertexArray(attrMap, ibo);
    } else {
      itr.second.vaoPtr = nullptr;
    }

    if (itr.second.strokeShaderPtr) {
      currRenderer->bindShader(itr.second.strokeShaderPtr);

      VboAttrToShaderAttrMap strokeAttrMap;
      ::Rendering::GL::Resources::GLIndexBufferShPtr strokeIbo;
      _buildVAOData(itr.first, itr.second.strokeShaderPtr.get(), strokeAttrMap, strokeIbo);

      itr.second.strokeVaoPtr = rsrcMgr->createVertexArray(strokeAttrMap, strokeIbo);
    } else {
      itr.second.strokeVaoPtr = nullptr;
    }
  }

  // if (currRenderer && prevRenderer != currRenderer) {
  //   if (prevRenderer) {
  //     prevRenderer->makeActiveOnCurrentThread(prevWindow);
  //   } else {
  //     currRenderer->makeInactive();
  //   }
  // }

  _propsDirty = false;
}

std::set<GpuId> BaseMark::_initUnusedGpus() const {
  std::set<GpuId> unusedGpus;
  for (const auto& kv : _perGpuData) {
    unusedGpus.insert(kv.first);
  }
  return unusedGpus;
}

void BaseMark::_initGpuResources(const QueryRendererContext* ctx, bool initializing) {
  auto qrmPerGpuDataPtr = ctx->getRootGpuCache()->perGpuData;
  CHECK(qrmPerGpuDataPtr);

  ::Rendering::GL::GLRenderer* renderer;
  ::Rendering::GL::GLResourceManagerShPtr rsrcMgr;

  int numGpus = _perGpuData.size();
  bool update = (numGpus > 0 && (_perGpuData.begin()->second.shaderPtr || _perGpuData.begin()->second.strokeShaderPtr));
  bool createdNewGpuRsrc = false;

  std::set<GpuId> usedGpuIds;
  auto unusedGpus = _initUnusedGpus();

  auto qrmItr = qrmPerGpuDataPtr->begin();

  if (!_dataPtr) {
    usedGpuIds.insert((*qrmItr)->gpuId);
    auto perGpuItr = _perGpuData.find((*qrmItr)->gpuId);
    if (perGpuItr == _perGpuData.end()) {
      PerGpuData gpuData(*qrmItr);
      if (update) {
        auto beginItr = _perGpuData.begin();
        CHECK(beginItr != _perGpuData.end());

        beginItr->second.makeActiveOnCurrentThread();

        std::string vertSrc, fragSrc, strokeVertSrc, strokeFragSrc, strokeGeomSrc;

        if (beginItr->second.shaderPtr) {
          vertSrc = beginItr->second.shaderPtr->getVertexSource();
          fragSrc = beginItr->second.shaderPtr->getFragmentSource();
        }

        if (beginItr->second.strokeShaderPtr) {
          strokeVertSrc = beginItr->second.strokeShaderPtr->getVertexSource();
          strokeGeomSrc = beginItr->second.strokeShaderPtr->getGeometrySource();
          strokeFragSrc = beginItr->second.strokeShaderPtr->getFragmentSource();
        }

        (*qrmItr)->makeActiveOnCurrentThread();
        renderer = (*qrmItr)->getGLRenderer();
        CHECK(renderer);
        rsrcMgr = renderer->getResourceManager();

        if (beginItr->second.shaderPtr) {
          // TODO(croot): make resource copy constructors which appropriately
          // deal with different contexts, including contexts on different gpus
          gpuData.shaderPtr = rsrcMgr->createShader(vertSrc, fragSrc);
        }

        if (beginItr->second.strokeShaderPtr) {
          gpuData.strokeShaderPtr = rsrcMgr->createShader(strokeVertSrc, strokeFragSrc, strokeGeomSrc);
        }

        // NOTE: we need to create the VAO after the properties have
        // been updated.
        createdNewGpuRsrc = true;
      }
      _perGpuData.emplace((*qrmItr)->gpuId, std::move(gpuData));
    }
    unusedGpus.erase((*qrmItr)->gpuId);
  } else {
    usedGpuIds = _dataPtr->getUsedGpuIds();

    for (auto& gpuId : usedGpuIds) {
      auto perGpuItr = _perGpuData.find(gpuId);
      if (perGpuItr == _perGpuData.end()) {
        qrmItr = qrmPerGpuDataPtr->find(gpuId);
        CHECK(qrmItr != qrmPerGpuDataPtr->end());

        PerGpuData gpuData(*qrmItr);
        if (update) {
          auto beginItr = _perGpuData.begin();
          CHECK(beginItr != _perGpuData.end() && beginItr->second.shaderPtr);

          beginItr->second.makeActiveOnCurrentThread();

          std::string vertSrc, fragSrc, strokeVertSrc, strokeFragSrc, strokeGeomSrc;

          if (beginItr->second.shaderPtr) {
            vertSrc = beginItr->second.shaderPtr->getVertexSource();
            fragSrc = beginItr->second.shaderPtr->getFragmentSource();
          }

          if (beginItr->second.strokeShaderPtr) {
            strokeVertSrc = beginItr->second.strokeShaderPtr->getVertexSource();
            strokeGeomSrc = beginItr->second.strokeShaderPtr->getGeometrySource();
            strokeFragSrc = beginItr->second.strokeShaderPtr->getFragmentSource();
          }

          (*qrmItr)->makeActiveOnCurrentThread();
          renderer = (*qrmItr)->getGLRenderer();
          CHECK(renderer);

          rsrcMgr = renderer->getResourceManager();

          if (beginItr->second.shaderPtr) {
            // TODO(croot): make resource copy constructors which appropriately
            // deal with different contexts, including contexts on different gpus
            gpuData.shaderPtr = rsrcMgr->createShader(vertSrc, fragSrc);
          }

          if (beginItr->second.strokeShaderPtr) {
            gpuData.strokeShaderPtr = rsrcMgr->createShader(strokeVertSrc, strokeFragSrc, strokeGeomSrc);
          }

          // NOTE: we need to create the VAO after the properties have
          // been updated.
          createdNewGpuRsrc = true;
        }
        _perGpuData.emplace((*qrmItr)->gpuId, std::move(gpuData));
      }

      unusedGpus.erase(gpuId);
    }
  }

  for (auto gpuId : unusedGpus) {
    _perGpuData.erase(gpuId);
  }

  if (numGpus && _perGpuData.size() == 0) {
    // TODO(croot): make a makeAllDirty() function
    setShaderDirty();
    setPropsDirty();
  }

  // key.initGpuResources(ctx, unusedGpus, initializing);
  key.initGpuResources(ctx, usedGpuIds, unusedGpus);

  if (!initializing) {
    _updateRenderPropertyGpuResources(ctx, usedGpuIds, unusedGpus);
  }

  if (createdNewGpuRsrc) {
    for (auto& itr : _perGpuData) {
      if (!itr.second.vaoPtr) {
        itr.second.makeActiveOnCurrentThread();
        renderer = itr.second.getGLRenderer();
        CHECK(renderer);

        rsrcMgr = renderer->getResourceManager();

        if (itr.second.shaderPtr) {
          renderer->bindShader(itr.second.shaderPtr);

          ::Rendering::GL::Resources::VboAttrToShaderAttrMap attrMap;
          ::Rendering::GL::Resources::GLIndexBufferShPtr ibo;
          _buildVAOData(itr.first, itr.second.shaderPtr.get(), attrMap, ibo);

          itr.second.vaoPtr = rsrcMgr->createVertexArray(attrMap);
        } else {
          itr.second.vaoPtr = nullptr;
        }

        if (itr.second.strokeShaderPtr) {
          VboAttrToShaderAttrMap strokeAttrMap;
          ::Rendering::GL::Resources::GLIndexBufferShPtr strokeIbo;
          _buildVAOData(itr.first, itr.second.strokeShaderPtr.get(), strokeAttrMap, strokeIbo);

          itr.second.strokeVaoPtr = rsrcMgr->createVertexArray(strokeAttrMap, strokeIbo);
        } else {
          itr.second.strokeVaoPtr = nullptr;
        }
      }
    }
  }
}

}  // namespace QueryRenderer
