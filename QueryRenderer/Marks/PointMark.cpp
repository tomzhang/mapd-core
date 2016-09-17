#include "PointMark.h"
#include "../Data/QueryDataTable.h"
#include "../Scales/ScaleRef.h"
#include "shaders/pointTemplate_vert.h"
#include "shaders/pointTemplate_frag.h"
#include "../Utils/ShaderUtils.h"

#include <Rendering/Renderer/GL/GLResourceManager.h>

#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/regex.hpp>

// #include <fstream>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;
using ::Rendering::GL::Resources::GLShader;
using ::Rendering::GL::Resources::VboLayoutAttrToShaderAttrMap;

PointMark::PointMark(const rapidjson::Value& obj,
                     const rapidjson::Pointer& objPath,
                     const QueryRendererContextShPtr& ctx)
    : BaseMark(GeomType::POINTS, ctx, obj, objPath, QueryDataTableBaseType::BASIC_VBO, false),
      x(this, "x", ctx),
      y(this, "y", ctx),
      z(this, "z", ctx),
      size(this, "size", ctx),

      // TODO(croot): let's log a warning and continue onwards if
      // hit testing is asked for, but the input sql data doesn't
      // have an id.
      id(this, "id", ctx, false),
      fillColor(this, "fillColor", ctx, true, true, true) {
  _initPropertiesFromJSONObj(obj, objPath);
  _jsonPath = objPath;
  _updateShader();
}

PointMark::~PointMark() {
}

std::set<BaseRenderProperty*> PointMark::_getUsedProps() {
  std::set<BaseRenderProperty*> rtn = {&x, &y, &size, &fillColor};  // TODO(croot) add z
  if (_ctx->doDepthTest()) {
    rtn.insert(&z);
  }
  if (_ctx->doHitTest()) {
    rtn.insert(&id);
  }
  return rtn;
}

void PointMark::_initPropertiesFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  // TODO(croot): move "properties" to a const somewhere
  std::string propertiesProp = "properties";

  rapidjson::Value::ConstMemberIterator mitr;

  RUNTIME_EX_ASSERT((mitr = obj.FindMember(propertiesProp.c_str())) != obj.MemberEnd(),
                    RapidJSONUtils::getJsonParseErrorStr(
                        _ctx->getUserWidgetIds(), obj, "mark objects must have a \"properties\" property."));

  const rapidjson::Value& propObj = mitr->value;

  if (!_ctx->isJSONCacheUpToDate(_propertiesJsonPath, propObj)) {
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
            _ctx->getUserWidgetIds(), propObj, "\"" + xProp + "\" mark property must exist for point marks."));

    if (!_ctx->isJSONCacheUpToDate(_xJsonPath, mitr->value)) {
      _xJsonPath = _propertiesJsonPath.Append(xProp.c_str(), xProp.length());
      RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsNumber()),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(),
                            mitr->value,
                            "\"" + xProp + "\" mark property must be a scale/data reference or a number."));
      x.initializeFromJSONObj(mitr->value, _xJsonPath, _dataPtr);
    } else {
      _xJsonPath = _propertiesJsonPath.Append(xProp.c_str(), xProp.length());
    }

    // TODO(croot): move "y" to a const somewhere
    std::string yProp = "y";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(yProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + yProp + "\" mark property must exist for point marks."));

    if (!_ctx->isJSONCacheUpToDate(_yJsonPath, mitr->value)) {
      _yJsonPath = _propertiesJsonPath.Append(yProp.c_str(), yProp.length());
      RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsNumber()),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(),
                            mitr->value,
                            "\"" + yProp + "\" mark property must be a scale/data reference or a number."));
      y.initializeFromJSONObj(mitr->value, _yJsonPath, _dataPtr);
    } else {
      _yJsonPath = _propertiesJsonPath.Append(yProp.c_str(), yProp.length());
    }

    // TODO(croot): move "z" to a const somewhere
    std::string zProp = "z";
    if ((mitr = propObj.FindMember(zProp.c_str())) != propObj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_zJsonPath, mitr->value)) {
        _zJsonPath = _propertiesJsonPath.Append(zProp.c_str(), zProp.length());
        RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsNumber()),
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(),
                              mitr->value,
                              "\"" + zProp + "\" mark property must be a scale/data reference or a number."));
        z.initializeFromJSONObj(mitr->value, _zJsonPath, _dataPtr);
      } else {
        _zJsonPath = _propertiesJsonPath.Append(zProp.c_str(), zProp.length());
      }
    } else {
      // empty the json path for z
      _zJsonPath = rapidjson::Pointer();
    }

    // TODO(croot): move "size" to a const somewhere
    std::string sizeProp = "size";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(sizeProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + sizeProp + "\" mark property must exist for point marks."));

    if (!_ctx->isJSONCacheUpToDate(_sizeJsonPath, mitr->value)) {
      _sizeJsonPath = _propertiesJsonPath.Append(sizeProp.c_str(), sizeProp.length());
      RUNTIME_EX_ASSERT((mitr->value.IsObject() || mitr->value.IsNumber()),
                        RapidJSONUtils::getJsonParseErrorStr(
                            _ctx->getUserWidgetIds(),
                            mitr->value,
                            "\"" + sizeProp + "\" mark property must be a scale/data reference or a number."));
      size.initializeFromJSONObj(mitr->value, _sizeJsonPath, _dataPtr);
    } else {
      _sizeJsonPath = _propertiesJsonPath.Append(sizeProp.c_str(), sizeProp.length());
    }

    // TODO(croot): move "fillColor" to a const somewhere
    std::string fillColorProp = "fillColor";
    RUNTIME_EX_ASSERT(
        (mitr = propObj.FindMember(fillColorProp.c_str())) != propObj.MemberEnd(),
        RapidJSONUtils::getJsonParseErrorStr(
            _ctx->getUserWidgetIds(), propObj, "\"" + fillColorProp + "\" mark property must exist for point marks."));

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
    } else {
      // clear out id path
      _idJsonPath = rapidjson::Pointer();
    }

    _updateProps(_getUsedProps());
  } else {
    _propertiesJsonPath = objPath.Append(propertiesProp.c_str(), propertiesProp.length());
  }
}

void PointMark::_updateShader() {
  // TODO(croot): need to determine a build-appropriate way to access
  // shaders. The best way probably is to create a cmake build
  // script that converts all shaders into their own header
  // files with static strings of the shader's source to access.

  if (!_shaderDirty || !_perGpuData.size()) {
    // early out
    return;
  }

  std::string vertSrc(PointTemplate_Vert::source);

  std::vector<BaseRenderProperty*> props = {&x, &y, &size, &fillColor};  // TODO: add z & fillColor

  bool useKey = key.hasVboPtr();
  boost::replace_first(vertSrc, "<useKey>", std::to_string(useKey));

  // update all the types first
  for (auto& prop : props) {
    ShaderUtils::setRenderPropertyTypeInShaderSrc(*prop, vertSrc);
  }

  // now set props as uniform or vertex attrs
  for (auto& prop : _vboProps) {
    ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(*prop, vertSrc, false);
  }

  for (auto& prop : _uniformProps) {
    ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(*prop, vertSrc, true);
  }

  if (_ctx->doHitTest()) {
    props.push_back(&id);
  } else {
    // define the id as uniform to get the shader to compile
    ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(id, vertSrc, true);
  }

  // now insert any additional functionality
  // std::unordered_map<std::string, BaseScale*> visitedScales;
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
      std::string suffix = "_" + prop->getName();
      propFuncName = prop->getGLSLFunc();

      funcRange = ShaderUtils::getGLSLFunctionBounds(vertSrc, propFuncName);

      RUNTIME_EX_ASSERT(!funcRange.empty(),
                        std::string(*this) + ": Cannot find a properly defined \"" + propFuncName +
                            "\" function in the vertex shader.");

      std::string scaleCode = scalePtr->getGLSLCode(suffix);

      boost::replace_range(vertSrc, funcRange, scaleCode);

      funcName = scalePtr->getScaleGLSLFuncName(suffix);

      if (*prop->getInTypeGL() == *scalePtr->getDomainTypeGL()) {
        boost::replace_all(vertSrc, propFuncName + "(" + prop->getName() + ")", funcName + "(" + prop->getName() + ")");

      } else {
        std::string domainType = scalePtr->getDomainGLSLTypeName(suffix);
        boost::replace_all(vertSrc,
                           propFuncName + "(" + prop->getName() + ")",
                           funcName + "(" + domainType + "(" + prop->getName() + "))");
      }
    }
  }

  std::string fragSrc(PointTemplate_Frag::source);

  BaseMark::_updateShader(vertSrc, fragSrc);

  // static int CROOTcnt = 0;
  // CROOTcnt++;
  // if (CROOTcnt == 1) {
  //   std::ofstream shadersrcstream;
  //   shadersrcstream.open("shadersource.vert");
  //   shadersrcstream << vertSrc;
  //   shadersrcstream.close();

  //   shadersrcstream.open("shadersource.frag");
  //   shadersrcstream << fragSrc;
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

void PointMark::_buildVAOData(const GpuId& gpuId,
                              ::Rendering::GL::Resources::GLShader* activeShader,
                              VboLayoutAttrToShaderAttrMap& attrMap,
                              ::Rendering::GL::Resources::GLIndexBufferShPtr& ibo) {
  int cnt = 0;
  int vboSize = 0;
  int itrSize = 0;
  for (auto& itr : _vboProps) {
    if (!activeShader->hasVertexAttribute(itr->getName())) {
      continue;
    }

    cnt++;
    itrSize = itr->size(gpuId);
    if (cnt == 1) {
      vboSize = itrSize;
    } else {
      RUNTIME_EX_ASSERT(itrSize == vboSize,
                        std::string(*this) +
                            ": Invalid point mark. The sizes of the vertex buffer attributes do not match for gpuId " +
                            std::to_string(gpuId) + ". " + std::to_string(vboSize) + "!=" + std::to_string(itrSize));
    }

    itr->addToVboAttrMap(gpuId, attrMap);
    // itr->bindToRenderer(activeShader);
  }
}

void PointMark::_bindUniformProperties(GLShader* activeShader) {
  std::unordered_map<std::string, std::string> subroutines;

  // TODO(croot): create a static invalidKeyAttrName string on the class
  static const std::string invalidKeyAttrName = "invalidKey";
  if (key.hasVboPtr()) {
    if (activeShader->hasUniformAttribute(invalidKeyAttrName)) {
      GLint type = activeShader->getUniformAttributeGLType(invalidKeyAttrName);
      if (type == GL_INT) {
        auto vboPtr = key.getVboPtr();
        CHECK(vboPtr);
        auto queryResultBuffer = dynamic_cast<QueryResultVertexBuffer*>(vboPtr.get());
        if (queryResultBuffer) {
          auto queryDataLayoutPtr = queryResultBuffer->getQueryDataLayout();
          activeShader->setUniformAttribute<int>(invalidKeyAttrName, static_cast<int>(queryDataLayoutPtr->invalidKey));
        }
      }  // else if (GLEW_NV_vertex_attrib_integer_64bit && type == GL_INT64_NV) {
         // TODO(croot) - do we need to do the glew extension check above or
         // would there be an error at shader compilation if the extension
         // didn't exist?

      // TODO(croot) fill this out
      // }
    }
  }

  for (auto prop : _vboProps) {
    const ScaleRefShPtr& scalePtr = prop->getScaleReference();
    if (scalePtr != nullptr) {
      scalePtr->bindUniformsToRenderer(activeShader, subroutines, "_" + prop->getName());
    }
  }

  for (auto prop : _uniformProps) {
    const ScaleRefShPtr& scalePtr = prop->getScaleReference();
    bool hasDensityAccumulator = false;
    if (scalePtr != nullptr) {
      scalePtr->bindUniformsToRenderer(activeShader, subroutines, "_" + prop->getName());
      hasDensityAccumulator = (scalePtr->getAccumulatorType() == AccumulatorType::DENSITY);
    }

    if (!hasDensityAccumulator || activeShader->hasUniformAttribute(prop->getName())) {
      prop->bindUniformToRenderer(activeShader, prop->getName());
    }
  }

  // TODO(croot): fold this into the base class
  if (_ctx->doHitTest() && _dataPtr) {
    auto dataPtr = std::dynamic_pointer_cast<BaseQueryDataTableSQL>(_dataPtr);
    if (dataPtr) {
      activeShader->setUniformAttribute("uTableId", dataPtr->getTableId());
    }
  }

  if (subroutines.size()) {
    activeShader->setSubroutines(subroutines);
  }
}

void PointMark::_updateRenderPropertyGpuResources(const QueryRendererContext* ctx,
                                                  const std::set<GpuId>& usedGpus,
                                                  const std::set<GpuId>& unusedGpus) {
  x.initGpuResources(ctx, usedGpus, unusedGpus);
  y.initGpuResources(ctx, usedGpus, unusedGpus);
  z.initGpuResources(ctx, usedGpus, unusedGpus);
  size.initGpuResources(ctx, usedGpus, unusedGpus);
  id.initGpuResources(ctx, usedGpus, unusedGpus);
  fillColor.initGpuResources(ctx, usedGpus, unusedGpus);
}

void PointMark::draw(GLRenderer* renderer, const GpuId& gpuId) {
  // NOTE: shader should have been updated before calling this
  auto itr = _perGpuData.find(gpuId);
  if (itr == _perGpuData.end()) {
    return;
  }

  RootPerGpuDataShPtr qrmGpuData = itr->second.getRootPerGpuData();
  CHECK(qrmGpuData);

  ::Rendering::Renderer* rndr = qrmGpuData->rendererPtr.get();
  CHECK(itr->second.shaderPtr && rndr == renderer);

  // now bind the shader
  renderer->bindShader(itr->second.shaderPtr);
  renderer->bindVertexArray(itr->second.vaoPtr);

  _bindUniformProperties(itr->second.shaderPtr.get());

  // TODO(croot): render state stack -- push/pop
  renderer->enable(GL_PROGRAM_POINT_SIZE);

  if (!hasAccumulator()) {
    // Because we're doing a "discard" in the point shader,
    // we can benefit from turning on GL_SAMPLE_SHADING so
    // that all multi-samples are evaluated in the shader.
    // TODO(croot): evaluating the frag shader for every
    // sample can be constly, so look into only doing so
    // if points can be small, i.e. < 3 pixels wide. That's
    // where we get the most benefit from GL_SAMPLE_SHADING.
    renderer->enable(GL_SAMPLE_SHADING);
    renderer->setMinSampleShading(1.0);
  }

  // now draw points
  // auto dataPtr = std::dynamic_pointer_cast<BaseQueryDataTableSQLJSON>(_dataPtr);
  renderer->drawVertexBuffers(GL_POINTS, 0, itr->second.vaoPtr->numVertices());

  // unset state
  renderer->disable(GL_PROGRAM_POINT_SIZE);
  renderer->disable(GL_SAMPLE_SHADING);
}

bool PointMark::updateFromJSONObj(const rapidjson::Value& obj, const rapidjson::Pointer& objPath) {
  bool rtn = false;
  if (!_ctx->isJSONCacheUpToDate(_jsonPath, obj)) {
    BaseMark::_initFromJSONObj(obj, objPath, QueryDataTableBaseType::BASIC_VBO, false, false);
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

PointMark::operator std::string() const {
  return "PointMark " + to_string(_ctx->getUserWidgetIds());
}

}  // namespace QueryRenderer
