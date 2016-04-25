#include "BaseMark.h"
#include <Rendering/Renderer/GL/GLResourceManager.h>

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
      _uniformProps() {
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

void BaseMark::_buildVertexArrayObjectFromProperties() {
  if (!_propsDirty || !_perGpuData.size()) {
    // early out
    return;
  }

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
    CHECK(itr.second.shaderPtr != nullptr);

    currRenderer->bindShader(itr.second.shaderPtr);

    // build property map for how vertex buffer attributes will
    // be bound to shader attributes
    VboAttrToShaderAttrMap attrMap;
    _addPropertiesToAttrMap(itr.first, attrMap);

    GLResourceManagerShPtr rsrcMgr = currRenderer->getResourceManager();

    itr.second.vaoPtr = rsrcMgr->createVertexArray(attrMap);
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
  bool update = (numGpus > 0 && _perGpuData.begin()->second.shaderPtr);
  bool createdNewGpuRsrc = false;

  auto unusedGpus = _initUnusedGpus();

  auto qrmItr = qrmPerGpuDataPtr->begin();
  if (!_dataPtr) {
    auto perGpuItr = _perGpuData.find((*qrmItr)->gpuId);
    if (perGpuItr == _perGpuData.end()) {
      PerGpuData gpuData(*qrmItr);
      if (update) {
        auto beginItr = _perGpuData.begin();
        CHECK(beginItr != _perGpuData.end() && beginItr->second.shaderPtr);

        beginItr->second.makeActiveOnCurrentThread();
        std::string vertSrc = beginItr->second.shaderPtr->getVertexSource();
        std::string fragSrc = beginItr->second.shaderPtr->getFragmentSource();

        (*qrmItr)->makeActiveOnCurrentThread();
        renderer = (*qrmItr)->getGLRenderer();
        CHECK(renderer);

        rsrcMgr = renderer->getResourceManager();

        // TODO(croot): make resource copy constructors which appropriately
        // deal with different contexts, including contexts on different gpus
        gpuData.shaderPtr = rsrcMgr->createShader(vertSrc, fragSrc);

        // NOTE: we need to create the VAO after the properties have
        // been updated.
        createdNewGpuRsrc = true;
      }
      _perGpuData.emplace((*qrmItr)->gpuId, std::move(gpuData));
    }
    unusedGpus.erase((*qrmItr)->gpuId);
  } else {
    std::vector<GpuId> usedGpuIds = _dataPtr->getUsedGpuIds();

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
          std::string vertSrc = beginItr->second.shaderPtr->getVertexSource();
          std::string fragSrc = beginItr->second.shaderPtr->getFragmentSource();

          (*qrmItr)->makeActiveOnCurrentThread();
          renderer = (*qrmItr)->getGLRenderer();
          CHECK(renderer);

          rsrcMgr = renderer->getResourceManager();

          // TODO(croot): make resource copy constructors which appropriately
          // deal with different contexts, including contexts on different gpus
          gpuData.shaderPtr = rsrcMgr->createShader(vertSrc, fragSrc);

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
  key.initGpuResources(ctx, unusedGpus);

  if (!initializing) {
    _updateRenderPropertyGpuResources(ctx, unusedGpus);
  }

  if (createdNewGpuRsrc) {
    for (auto& itr : _perGpuData) {
      if (!itr.second.vaoPtr) {
        itr.second.makeActiveOnCurrentThread();
        renderer = itr.second.getGLRenderer();
        CHECK(renderer);

        rsrcMgr = renderer->getResourceManager();

        CHECK(itr.second.shaderPtr);
        renderer->bindShader(itr.second.shaderPtr);

        ::Rendering::GL::Resources::VboAttrToShaderAttrMap attrMap;
        _addPropertiesToAttrMap(itr.first, attrMap);

        itr.second.vaoPtr = rsrcMgr->createVertexArray(attrMap);
      }
    }
  }
}

}  // namespace QueryRenderer
