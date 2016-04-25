#include "PolyMark.h"
#include "shaders/polyTemplate_vert.h"
#include "shaders/polyTemplate_frag.h"
#include "../Utils/ShaderUtils.h"
#include "../Data/QueryPolyDataTable.h"
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>

#include <iostream>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;

PolyMark::PolyMark(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, const QueryRendererContextShPtr& ctx)
    : BaseMark(GeomType::POLYS, ctx, obj, objPath, QueryDataTableBaseType::POLY, true),
      x(this, "x", ctx),
      y(this, "y", ctx),
      // z(this, "z", ctx),

      fillColor(this, "fillColor", ctx),

      // TODO(croot): let's log a warning and continue onwards if
      // hit testing is asked for, but the input sql data doesn't
      // have an id.
      id(this, "id", ctx, false) {
  _initPropertiesFromJSONObj(obj, objPath);
  _updateShader();
}

PolyMark::~PolyMark() {
}

void PolyMark::_initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  // TODO(croot): move "properties" to a const somewhere
  std::string propertiesProp = "properties";

  rapidjson::Value::ConstMemberIterator mitr;

  RUNTIME_EX_ASSERT(
      (mitr = obj.FindMember(propertiesProp.c_str())) != obj.MemberEnd(),
      RapidJSONUtils::getJsonParseErrorStr(
          _ctx->getUserWidgetIds(), obj, "poly mark objects must have a \"" + propertiesProp + "\" property."));

  const rapidjson::Value& propObj = mitr->value;

  if (!_ctx->isJSONCacheUpToDate(_propertiesJsonPath, propObj)) {
    std::vector<BaseRenderProperty*> usedProps{&x, &y, &fillColor};  // TODO(croot) add z

    _propertiesJsonPath = objPath.Append(propertiesProp.c_str(), propertiesProp.length());

    RUNTIME_EX_ASSERT(
        propObj.IsObject(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "The \"properties\" property of marks must be a json object."));

    // TODO(croot): move "x" to a const somewhere
    std::string xProp = "x";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(xProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + xProp + "\" mark property must exist for poly marks."));

    if (!_ctx->isJSONCacheUpToDate(_xJsonPath, mitr->value)) {
      _xJsonPath = _propertiesJsonPath.Append(xProp.c_str(), xProp.length());
      RUNTIME_EX_ASSERT(
          mitr->value.IsObject(),
          RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                               mitr->value,
                                               "\"" + xProp + "\" property for polys must be a scale/data reference."));
      x.initializeFromJSONObj(mitr->value, _xJsonPath, _dataPtr);
    } else {
      _xJsonPath = _propertiesJsonPath.Append(xProp.c_str(), xProp.length());
    }

    // TODO(croot): move "y" to a const somewhere
    std::string yProp = "y";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(yProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + yProp + "\" mark property must exist for poly marks."));

    if (!_ctx->isJSONCacheUpToDate(_yJsonPath, mitr->value)) {
      _yJsonPath = _propertiesJsonPath.Append(yProp.c_str(), yProp.length());
      RUNTIME_EX_ASSERT(
          mitr->value.IsObject(),
          RapidJSONUtils::getJsonParseErrorStr(_ctx->getUserWidgetIds(),
                                               mitr->value,
                                               "\"" + yProp + "\" property for polys must be a scale/data reference."));
      y.initializeFromJSONObj(mitr->value, _yJsonPath, _dataPtr);
    } else {
      _yJsonPath = _propertiesJsonPath.Append(yProp.c_str(), yProp.length());
    }

    // TODO(croot): move "z" to a const somewhere
    // std::string zProp = "z";
    // if ((mitr = propObj.FindMember(zProp.c_str())) != propObj.MemberEnd()) {
    //   if (!_ctx->isJSONCacheUpToDate(_zJsonPath, mitr->value)) {
    //     _zJsonPath = _propertiesJsonPath.Append(zProp.c_str(), zProp.length());
    //     RUNTIME_EX_ASSERT(mitr->value.IsObject(),
    //                       RapidJSONUtils::getJsonParseErrorStr(
    //                           _ctx->getUserWidgetIds(),
    //                           mitr->value,
    //                           "\"" + zProp + "\" property for polys must be a scale/data reference."));
    //     z.initializeFromJSONObj(mitr->value, _zJsonPath, _dataPtr);
    //   } else {
    //     _zJsonPath = _propertiesJsonPath.Append(zProp.c_str(), zProp.length());
    //   }

    //   usedProps.push_back(&z);
    // } else {
    //   // empty the json path for z
    //   _zJsonPath = rapidjson::Pointer();
    // }

    // TODO(croot): move "fillColor" to a const somewhere
    std::string fillColorProp = "fillColor";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(fillColorProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + fillColorProp + "\" mark property must exist for poly marks."));

    if (!_ctx->isJSONCacheUpToDate(_fillColorJsonPath, mitr->value)) {
      _fillColorJsonPath = _propertiesJsonPath.Append(fillColorProp.c_str(), fillColorProp.length());
      RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsString()),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(),
                            mitr->value,
                            "\"" + fillColorProp + "\" mark property must be a scale/data reference or a string."));
      fillColor.initializeFromJSONObj(mitr->value, _fillColorJsonPath, _dataPtr);
    } else {
      _fillColorJsonPath = _propertiesJsonPath.Append(fillColorProp.c_str(), fillColorProp.length());
    }

    if (_ctx->doHitTest()) {
      // TODO(croot): move "id" to a const somewhere
      std::string idProp = "id";

      if ((mitr = propObj.FindMember(idProp.c_str())) != propObj.MemberEnd()) {
        if (!_ctx->isJSONCacheUpToDate(_idJsonPath, mitr->value)) {
          _idJsonPath = _propertiesJsonPath.Append(idProp.c_str(), idProp.length());
          RUNTIME_EX_ASSERT(mitr->value.IsObject(),
                            RapidJSONUtils::getJsonParseErrorStr(
                                _ctx->getUserWidgetIds(),
                                mitr->value,
                                "\"id\" is a special mark property that must be defined by a data reference."));
          id.initializeFromJSONObj(mitr->value, _idJsonPath, _dataPtr);
        } else {
          // update the json path, if it's been changed
          _idJsonPath = _propertiesJsonPath.Append(idProp.c_str(), idProp.length());
        }
      } else {
        // clear out the json path
        _idJsonPath = rapidjson::Pointer();
        if (_dataPtr != nullptr) {
          id.initializeFromData(DataTable::defaultIdColumnName, _dataPtr);
        } else {
          id.initializeValue(0);  // reaching here "should" guarantee that there's only
                                  // 1 row of data
        }
      }

      usedProps.push_back(&id);
    } else {
      // clear out id path
      _idJsonPath = rapidjson::Pointer();
    }

    // Now update which props are vbo-defined, and which will be uniforms
    _vboProps.clear();
    _uboProps.clear();
    _uniformProps.clear();

    for (const auto& prop : usedProps) {
      if (prop->hasVboPtr()) {
        _vboProps.push_back(prop);
      } else if (prop->hasUboPtr()) {
        _uboProps.push_back(prop);
      } else {
        _uniformProps.push_back(prop);
      }
    }
  } else {
    _propertiesJsonPath = objPath.Append(propertiesProp.c_str(), propertiesProp.length());
  }
}

void PolyMark::_updateShader() {
  if (!_shaderDirty || !_perGpuData.size()) {
    // early out
    return;
  }

  std::string vertSrc(PolyTemplate_Vert::source);

  std::vector<BaseRenderProperty*> props = {&x, &y, &fillColor};  // TODO: add z & fillColor

  // update all the types first
  for (auto& prop : props) {
    ShaderUtils::setRenderPropertyTypeInShaderSrc(*prop, vertSrc);
  }

  // TODO(croot): support per vertex colors?
  bool usePerVertColor = false;
  boost::replace_first(vertSrc, "<usePerVertColor>", std::to_string(usePerVertColor));

  bool useUniformBuffer = (_uboProps.size() > 0);
  boost::replace_first(vertSrc, "<useUniformBuffer>", std::to_string(useUniformBuffer));

  for (auto& prop : _vboProps) {
    ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(*prop, vertSrc, false);
  }

  for (auto& prop : _uboProps) {
    ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(*prop, vertSrc, false);
  }

  for (auto& prop : _uniformProps) {
    ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(*prop, vertSrc, true);
  }

  bool useUniformId = true;
  if (_ctx->doHitTest()) {
    props.push_back(&id);
    useUniformId = false;
  }
  ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(id, vertSrc, useUniformId);

  QueryUniformBufferShPtr ubo;
  if (useUniformBuffer) {
    // double check that shader block layout that the uniform buffer
    // uses matches that in the shader
    CHECK(_dataPtr && _dataPtr->getBaseType() == QueryDataTableBaseType::POLY);
    ubo = (_perGpuData.size() ? _uboProps[0]->getUboPtr(_perGpuData.begin()->first) : nullptr);
    CHECK(ubo);

    auto shaderBlockLayout = ubo->getGLUniformBufferPtr()->getBufferLayout();

    std::string shaderBlockCode = shaderBlockLayout->buildShaderBlockCode(
        "PolyData", "polyData", ::Rendering::GL::Resources::StorageQualifier::UNIFORM);

    std::string propName;
    std::string colName;
    for (auto& prop : _uboProps) {
      propName = prop->getName();
      colName = prop->getDataColumnName();
      CHECK(colName.length() > 0) << "prop " << propName << " isn't initialized with buffer data";

      boost::replace_first(shaderBlockCode, colName, propName);
    }

    boost::replace_first(vertSrc, "<polyData>", shaderBlockCode);
  }

  // now insert any additional functionality
  std::unordered_map<std::string, BaseScale*>::iterator itr;

  std::string funcName;
  std::string propFuncName;
  ShaderUtils::str_itr_range funcRange;

  for (auto prop : props) {
    const ScaleRefShPtr& scalePtr = prop->getScaleReference();
    if (scalePtr != nullptr) {
      // NOTE: Because the domains of scales can be coerced into
      // the render property's type, we need to provide a new
      // set of GLSL code for each scale reference, even tho
      // it is possible to reference the same scale multiple times.

      // TODO(croot): there are ways we can reduce the amount of
      // shader code here. Domains of certain scales can be coerced,
      // but not all scales, so we can find some optimizations there.
      // Also, ranges can not be coerced, so optimizations can be
      // do there as well, but it is likely rare that the same
      // scale be referenced many times at this point (11/9/15), so
      // it's probably not worth the effort to optimize at this point.
      propFuncName = prop->getGLSLFunc();

      funcRange = ShaderUtils::getGLSLFunctionBounds(vertSrc, propFuncName);

      RUNTIME_EX_ASSERT(!funcRange.empty(),
                        std::string(*this) + ": Cannot find a properly defined \"" + propFuncName +
                            "\" function in the vertex shader.");

      std::string scaleCode = scalePtr->getGLSLCode("_" + prop->getName());

      boost::replace_range(vertSrc, funcRange, scaleCode);

      funcName = scalePtr->getScaleGLSLFuncName("_" + prop->getName());

      boost::replace_all(
          vertSrc, prop->getGLSLFunc() + "(" + prop->getName() + ")", funcName + "(" + prop->getName() + ")");

      boost::replace_all(vertSrc,
                         prop->getGLSLFunc() + "(polyData." + prop->getName() + ")",
                         funcName + "(polyData." + prop->getName() + ")");
    }
  }

  // std::string fragSrc = getShaderCodeFromFile(pointFragmentShaderFilename);
  std::string fragSrc(PolyTemplate_Frag::source);

  boost::replace_first(fragSrc, "<usePerVertColor>", std::to_string(usePerVertColor));

  // static int CROOTcnt = 0;
  // CROOTcnt++;
  // if (CROOTcnt == 1) {
  //   std::ofstream shadersrcstream;
  //   shadersrcstream.open("shadersource.vert");
  //   shadersrcstream << vertSrc;
  //   shadersrcstream.close();
  // }

  // now build the shader object
  // TODO(croot): How would we share shaders across different
  // query renderers?

  // TODO(croot): Make thread safe?

  GLRenderer* prevRenderer = GLRenderer::getCurrentThreadRenderer();
  GLRenderer* currRenderer = nullptr;
  RootPerGpuDataShPtr qrmGpuData;
  for (auto& itr : _perGpuData) {
    qrmGpuData = itr.second.getRootPerGpuData();
    qrmGpuData->makeActiveOnCurrentThread();
    currRenderer = dynamic_cast<GLRenderer*>(qrmGpuData->rendererPtr.get());

    GLResourceManagerShPtr rsrcMgr = currRenderer->getResourceManager();
    itr.second.shaderPtr = rsrcMgr->createShader(vertSrc, fragSrc);

    // if (useUniformBuffer) {
    //   // NOTE: only need to check the shader block layout of only the
    //   // first ubo as it will be the same for all other gpu ubos
    //   CHECK(*ubo->getGLUniformBufferPtr()->getBufferLayout().get() ==
    //         *itr.second.shaderPtr->getBlockLayout("PolyData"));
    // }

    // TODO(croot): should I make make the current thread
    // have an inactive renderer?
  }

  if (currRenderer && prevRenderer != currRenderer) {
    if (prevRenderer) {
      prevRenderer->makeActiveOnCurrentThread();
    } else {
      currRenderer->makeInactive();
    }
  }

  _shaderDirty = false;

  // set the props dirty to force a rebind with the new shader
  setPropsDirty();
}

void PolyMark::_addPropertiesToAttrMap(const GpuId& gpuId,
                                       ::Rendering::GL::Resources::VboAttrToShaderAttrMap& attrMap) {
  int cnt = 0;
  int vboSize = 0;
  int itrSize = 0;
  for (auto& itr : _vboProps) {
    cnt++;
    itrSize = itr->size(gpuId);
    if (cnt == 1) {
      vboSize = itrSize;
    } else {
      RUNTIME_EX_ASSERT(itrSize == vboSize,
                        std::string(*this) +
                            ": Invalid poly mark. The sizes of the vertex buffer attributes do not match for gpuId " +
                            std::to_string(gpuId) + ". " + std::to_string(vboSize) + "!=" + std::to_string(itrSize));
    }

    itr->addToVboAttrMap(gpuId, attrMap);
  }
}

void PolyMark::_bindUniformProperties(::Rendering::GL::Resources::GLShader* activeShader) {
  //   // TODO(croot): create a static invalidKeyAttrName string on the class
  // static const std::string invalidKeyAttrName = "invalidKey";
  // if (key.hasVboPtr()) {
  //   if (activeShader->hasUniformAttribute(invalidKeyAttrName)) {
  //     GLint type = activeShader->getUniformAttributeGLType(invalidKeyAttrName);
  //     if (type == GL_INT) {
  //       activeShader->setUniformAttribute<int>(invalidKeyAttrName, static_cast<int>(_invalidKey));
  //     }  // else if (GLEW_NV_vertex_attrib_integer_64bit && type == GL_INT64_NV) {
  //        // TODO(croot) - do we need to do the glew extension check above or
  //        // would there be an error at shader compilation if the extension
  //        // didn't exist?

  //     // TODO(croot) fill this out
  //     // }
  //   }
  // }

  for (auto prop : _vboProps) {
    const ScaleRefShPtr& scalePtr = prop->getScaleReference();
    if (scalePtr != nullptr) {
      scalePtr->bindUniformsToRenderer(activeShader, "_" + prop->getName());
    }
  }

  for (auto prop : _uboProps) {
    const ScaleRefShPtr& scalePtr = prop->getScaleReference();
    if (scalePtr != nullptr) {
      scalePtr->bindUniformsToRenderer(activeShader, "_" + prop->getName());
    }
  }

  for (auto prop : _uniformProps) {
    const ScaleRefShPtr& scalePtr = prop->getScaleReference();
    if (scalePtr != nullptr) {
      scalePtr->bindUniformsToRenderer(activeShader, "_" + prop->getName());
    }

    prop->bindUniformToRenderer(activeShader, prop->getName());
  }
}

void PolyMark::_updateRenderPropertyGpuResources(const QueryRendererContext* ctx, const std::set<GpuId> unusedGpus) {
  x.initGpuResources(ctx, unusedGpus);
  y.initGpuResources(ctx, unusedGpus);
  // z.initGpuResources(ctx, unusedGpus);
  id.initGpuResources(ctx, unusedGpus);
  fillColor.initGpuResources(ctx, unusedGpus);
}

void PolyMark::draw(::Rendering::GL::GLRenderer* renderer, const GpuId& gpuId) {
  // NOTE: shader should have been updated before calling this
  auto itr = _perGpuData.find(gpuId);
  CHECK(itr != _perGpuData.end());

  RootPerGpuDataShPtr qrmGpuData = itr->second.getRootPerGpuData();
  CHECK(qrmGpuData);

  ::Rendering::Renderer* rndr = qrmGpuData->rendererPtr.get();
  CHECK(itr->second.shaderPtr && rndr == renderer);

  // now bind the shader
  renderer->bindShader(itr->second.shaderPtr);
  renderer->bindVertexArray(itr->second.vaoPtr);
  _bindUniformProperties(itr->second.shaderPtr.get());

  CHECK(_dataPtr);
  QueryPolyDataTableShPtr polyTable = std::dynamic_pointer_cast<BaseQueryPolyDataTable>(_dataPtr);
  CHECK(polyTable);

  ::Rendering::GL::Resources::GLUniformBufferShPtr ubo;
  if (_uboProps.size()) {
    // the same ubo should be used for all ubo props, so only need to grab
    // from first one
    ubo = _uboProps[0]->getUboPtr(gpuId)->getGLUniformBufferPtr();
  }

  ::Rendering::GL::Resources::GLIndexBufferShPtr ibo = polyTable->getGLIndexBuffer(gpuId);
  ::Rendering::GL::Resources::GLIndirectDrawIndexBufferShPtr indibo = polyTable->getGLIndirectDrawIndexBuffer(gpuId);
  CHECK(ibo && indibo && (!ubo || indibo->numItems() == ubo->numItems()));

  renderer->bindIndexBuffer(ibo);
  renderer->bindIndirectDrawBuffer(indibo);
  if (ubo) {
    renderer->bindUniformBuffer(ubo);
    for (size_t i = 0; i < indibo->numItems(); ++i) {
      itr->second.shaderPtr->bindUniformBufferToBlock("PolyData", ubo, i);
      renderer->drawIndirectIndexBuffers(GL_TRIANGLES, i, 1);
    }
  } else {
    renderer->drawIndirectIndexBuffers(GL_TRIANGLES);
  }

  // now draw polys
  // renderer->drawVertexBuffers(GL_POINTS, 0, itr->second.vaoPtr->numItems());
}

bool PolyMark::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  bool rtn = false;
  if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    BaseMark::_initFromJSONObj(obj, objPath, QueryDataTableBaseType::POLY, false, false);
    _initPropertiesFromJSONObj(obj, objPath);
    rtn = true;
  } else if (_jsonPath != objPath) {
    // TODO(croot) - Bug! What if the cache is up-to-date, but the path has changed -- we need to update
    // the paths for this and all sub-objects that manage paths.
    // NOTE: We should never get in here since marks
    // are stored as an array but if we ever change the storage container
    // in the future (i.e. an unordered_map or the like), we'd reach this
    THROW_RUNTIME_EX(std::string(*this) + ": The cache for mark \"" + RapidJSONUtils::getPointerPath(objPath) +
                     "\" is up-to-date, but the path in the JSON has changed from " +
                     RapidJSONUtils::getPointerPath(_jsonPath) + " to " + RapidJSONUtils::getPointerPath(objPath) +
                     ", so the path caches need updating. This "
                     "has yet to be implemented.");
  }

  // TODO(croot): if the obj hasn't changed, but the path has, we need
  // to trickle that path change to all subobjects who cache their
  // json data. How should we handle this?
  _jsonPath = objPath;

  return rtn;
}

PolyMark::operator std::string() const {
  return "PolyMark " + to_string(_ctx->getUserWidgetIds());
}

}  // namespace QueryRenderer
