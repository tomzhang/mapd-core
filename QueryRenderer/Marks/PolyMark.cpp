#include "PolyMark.h"
#include "shaders/polyTemplate_vert.h"
#include "shaders/polyTemplate_frag.h"
#include "shaders/lineTemplate_vert.h"
#include "shaders/lineTemplate_geom.h"
#include "shaders/lineTemplate_frag.h"
#include "../Utils/ShaderUtils.h"
#include "../Data/QueryPolyDataTable.h"
#include <Rendering/Renderer/GL/GLRenderer.h>
#include <Rendering/Renderer/GL/GLResourceManager.h>
#include <regex>

namespace QueryRenderer {

using ::Rendering::GL::GLRenderer;
using ::Rendering::GL::GLResourceManagerShPtr;

PolyMark::PolyMark(const rapidjson::Value& obj, const rapidjson::Pointer& objPath, const QueryRendererContextShPtr& ctx)
    : BaseMark(GeomType::POLYS, ctx, obj, objPath, QueryDataTableBaseType::POLY, true),
      x(this, "x", ctx),
      y(this, "y", ctx),
      // z(this, "z", ctx),

      fillColor(this, "fillColor", ctx, true, false),
      strokeColor(this, "strokeColor", ctx, true, false),
      strokeWidth(this, "strokeWidth", ctx),
      lineJoin(this,
               "lineJoin",
               ctx,
               false,
               true,
               convertStringToLineJoinEnum),  // lineJoin, do not support scales at first
      miterLimit(this, "miterLimit", ctx, false),

      // TODO(croot): let's log a warning and continue onwards if
      // hit testing is asked for, but the input sql data doesn't
      // have an id.
      id(this, "id", ctx, false),
      _usedProps({&x, &y, &fillColor}),
      _usedStrokeProps({&x, &y, &strokeColor, &strokeWidth, &lineJoin, &miterLimit}) {
  if (_ctx->doHitTest()) {
    _usedProps.insert(&id);
    _usedStrokeProps.insert(&id);
  }
  // TODO(croot): add z to props when doDepthTest() is true

  _initPropertiesFromJSONObj(obj, objPath);
  _jsonPath = objPath;
  _updateShader();
}

PolyMark::~PolyMark() {}

std::set<BaseRenderProperty*> PolyMark::_getUsedProps() {
  std::set<BaseRenderProperty*> rtn = {
      &x, &y, &fillColor, &strokeColor, &strokeWidth, &lineJoin, &miterLimit};  // TODO(croot) add z
  if (_ctx->doHitTest()) {
    rtn.insert(&id);
  }
  return rtn;
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
      RUNTIME_EX_ASSERT(
          (mitr->value.IsObject() || mitr->value.IsString() || mitr->value.IsInt() || mitr->value.IsUint()),
          RapidJSONUtils::getJsonParseErrorStr(
              _ctx->getUserWidgetIds(),
              mitr->value,
              "\"" + fillColorProp +
                  "\" mark property must be a scale/data reference, a string, or packed into a 32-bit int/uint."));
      fillColor.initializeFromJSONObj(mitr->value, _fillColorJsonPath, _dataPtr);
    } else {
      _fillColorJsonPath = _propertiesJsonPath.Append(fillColorProp.c_str(), fillColorProp.length());
    }

    // TODO(croot): move "strokeWidth" to a const somewhere
    std::string strokeWidthProp = "strokeWidth";
    if ((mitr = propObj.FindMember(strokeWidthProp.c_str())) != propObj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_strokeWidthJsonPath, mitr->value)) {
        _strokeWidthJsonPath = _propertiesJsonPath.Append(strokeWidthProp.c_str(), strokeWidthProp.length());
        RUNTIME_EX_ASSERT(mitr->value.IsObject() || mitr->value.IsNumber(),
                          RapidJSONUtils::getJsonParseErrorStr(
                              _ctx->getUserWidgetIds(),
                              mitr->value,
                              "\"" + strokeWidthProp + "\" mark property must be a scale/data reference or a number."));
        strokeWidth.initializeFromJSONObj(mitr->value, _strokeWidthJsonPath, _dataPtr);
      } else {
        _strokeWidthJsonPath = _propertiesJsonPath.Append(strokeWidthProp.c_str(), strokeWidthProp.length());
      }
    } else {
      // set default strokeWidth to white
      // TODO(croot): expose strokeWidth default as a static somewhere
      strokeWidth.initializeValue(0);
      _strokeWidthJsonPath = rapidjson::Pointer();
    }

    // TODO(croot): move "lineJoin" to a const somewhere
    std::string lineJoinProp = "lineJoin";
    if ((mitr = propObj.FindMember(lineJoinProp.c_str())) != propObj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_lineJoinJsonPath, mitr->value)) {
        _lineJoinJsonPath = _propertiesJsonPath.Append(lineJoinProp.c_str(), lineJoinProp.length());
        RUNTIME_EX_ASSERT(
            mitr->value.IsInt() || mitr->value.IsUint64() || mitr->value.IsString(),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(),
                mitr->value,
                "\"" + lineJoinProp + "\" mark property must be an enum value (i.e. an int or a string)."));

        lineJoin.initializeFromJSONObj(mitr->value, _lineJoinJsonPath, _dataPtr);
      } else {
        _lineJoinJsonPath = _propertiesJsonPath.Append(lineJoinProp.c_str(), lineJoinProp.length());
      }
    } else {
      // set default lineJoin to white
      // TODO(croot): expose lineJoin default as a static somewhere
      lineJoin.initializeValue(static_cast<int>(LineJoinType::MITER));
      _lineJoinJsonPath = rapidjson::Pointer();
    }

    // TODO(croot): move "miterLimit" to a const somewhere
    std::string miterLimitProp = "miterLimit";
    if ((mitr = propObj.FindMember(miterLimitProp.c_str())) != propObj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_miterLimitJsonPath, mitr->value)) {
        _miterLimitJsonPath = _propertiesJsonPath.Append(miterLimitProp.c_str(), miterLimitProp.length());
        RUNTIME_EX_ASSERT(
            mitr->value.IsNumber(),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(), mitr->value, "\"" + miterLimitProp + "\" mark property must be a number."));

        miterLimit.initializeFromJSONObj(mitr->value, _miterLimitJsonPath, _dataPtr);
      } else {
        _miterLimitJsonPath = _propertiesJsonPath.Append(miterLimitProp.c_str(), miterLimitProp.length());
      }
    } else {
      // set default miterLimit to 10.0
      // TODO(croot): expose miterLimit default as a static somewhere
      miterLimit.initializeValue(10.0);
      _miterLimitJsonPath = rapidjson::Pointer();
    }

    // TODO(croot): move "strokeColor" to a const somewhere
    std::string strokeColorProp = "strokeColor";
    if ((mitr = propObj.FindMember(strokeColorProp.c_str())) != propObj.MemberEnd()) {
      if (!_ctx->isJSONCacheUpToDate(_strokeColorJsonPath, mitr->value)) {
        _strokeColorJsonPath = _propertiesJsonPath.Append(strokeColorProp.c_str(), strokeColorProp.length());
        RUNTIME_EX_ASSERT(
            (mitr->value.IsObject() || mitr->value.IsString() || mitr->value.IsInt() || mitr->value.IsUint()),
            RapidJSONUtils::getJsonParseErrorStr(
                _ctx->getUserWidgetIds(),
                mitr->value,
                "\"" + strokeColorProp +
                    "\" mark property must be a scale/data reference, a string, or packed into a 32-bit int/uint."));
        strokeColor.initializeFromJSONObj(mitr->value, _strokeColorJsonPath, _dataPtr);
      } else {
        _strokeColorJsonPath = _propertiesJsonPath.Append(strokeColorProp.c_str(), strokeColorProp.length());
      }
    } else {
      // set default strokeColor to white
      // TODO(croot): expose strokeColor default as a static somewhere
      strokeColor.initializeValue({255, 255, 255, 255});
      _strokeColorJsonPath = rapidjson::Pointer();
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

void PolyMark::_buildShaderSrc(std::vector<std::string>& shaderSrcs,
                               std::vector<BaseRenderProperty*>& props,
                               const std::string& uniformBlockName,
                               const std::string& uniformBlockInstanceName) {
  // update all the types first
  for (auto& prop : props) {
    for (auto& shaderSrc : shaderSrcs) {
      ShaderUtils::setRenderPropertyTypeInShaderSrc(*prop, shaderSrc);
    }
  }

  // TODO(croot): support per vertex colors?
  bool usePerVertColor = false;
  for (auto& shaderSrc : shaderSrcs) {
    boost::replace_first(shaderSrc, "<usePerVertColor>", std::to_string(usePerVertColor));
  }

  // TODO(croot): support per vertex widths?
  bool usePerVertWidth = false;
  for (auto& shaderSrc : shaderSrcs) {
    boost::replace_first(shaderSrc, "<usePerVertWidth>", std::to_string(usePerVertWidth));
  }

  bool useUniformBuffer = (_uboProps.size() > 0);
  for (auto& shaderSrc : shaderSrcs) {
    boost::replace_first(shaderSrc, "<useUniformBuffer>", std::to_string(useUniformBuffer));
  }

  for (auto& prop : _vboProps) {
    for (auto& shaderSrc : shaderSrcs) {
      ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(*prop, shaderSrc, false);
    }
  }

  for (auto& prop : _uboProps) {
    for (auto& shaderSrc : shaderSrcs) {
      ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(*prop, shaderSrc, false);
    }
  }

  for (auto& prop : _uniformProps) {
    for (auto& shaderSrc : shaderSrcs) {
      ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(*prop, shaderSrc, true);
    }
  }

  bool useUniformId = true;
  bool usePerVertId = false;
  if (_ctx->doHitTest()) {
    props.push_back(&id);
    useUniformId = false;
  }
  for (auto& shaderSrc : shaderSrcs) {
    ShaderUtils::setRenderPropertyAttrTypeInShaderSrc(id, shaderSrc, useUniformId);
    boost::replace_first(shaderSrc, "<usePerVertId>", std::to_string(usePerVertId));
  }

  QueryUniformBufferShPtr ubo;
  if (useUniformBuffer) {
    // double check the uniform buffer's shader block layout
    // matches that in the shader
    CHECK(_dataPtr && _dataPtr->getBaseType() == QueryDataTableBaseType::POLY);
    ubo = (_perGpuData.size() ? (*_uboProps.begin())->getUboPtr(_perGpuData.begin()->first) : nullptr);
    CHECK(ubo);

    auto glUbo = ubo->getGLUniformBufferPtr();
    CHECK(glUbo->getNumBufferLayouts() == 1);
    auto shaderBlockLayout = glUbo->getBufferLayoutAtIndex(0);

    std::string shaderBlockCode = shaderBlockLayout->buildShaderBlockCode(
        uniformBlockName, uniformBlockInstanceName, ::Rendering::GL::Resources::StorageQualifier::UNIFORM);

    std::string propName;
    std::string colName;
    for (auto& prop : _uboProps) {
      propName = prop->getName();
      colName = prop->getDataColumnName();
      CHECK(colName.length() > 0) << "prop " << propName << " isn't initialized with buffer data";

      for (auto& shaderSrc : shaderSrcs) {
        boost::replace_first(shaderSrc, "<" + uniformBlockInstanceName + ">", shaderBlockCode);
        boost::replace_all(
            shaderSrc, uniformBlockInstanceName + ".<" + propName + ">", uniformBlockInstanceName + "." + colName);
      }
    }
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
      std::string suffix = "_" + prop->getName();
      propFuncName = prop->getGLSLFunc();

      bool found = false;

      for (auto& shaderSrc : shaderSrcs) {
        if (!shaderSrc.length()) {
          continue;
        }

        funcRange = ShaderUtils::getGLSLFunctionBounds(shaderSrc, propFuncName);
        if (!funcRange.empty()) {
          found = true;
          std::string scaleCode = scalePtr->getGLSLCode(suffix);

          boost::replace_range(shaderSrc, funcRange, scaleCode);

          funcName = scalePtr->getScaleGLSLFuncName(suffix);

          if (*prop->getInTypeGL() == *scalePtr->getDomainTypeGL()) {
            boost::replace_all(shaderSrc, propFuncName + "(", funcName + "(");
          } else {
            std::string domainType = scalePtr->getDomainGLSLTypeName(suffix);
            std::regex funcRegex(propFuncName + "\\((\\s*[\\w.<>]*\\s*)\\)");

            shaderSrc = std::regex_replace(shaderSrc, funcRegex, funcName + "(" + domainType + "($1))");
          }
        }
      }

      RUNTIME_EX_ASSERT(
          found,
          std::string(*this) + ": Cannot find a properly defined \"" + propFuncName + "\" function in the shader.");
    }
  }

  // TODO(croot): Make thread safe?
}

bool PolyMark::_doFill() {
  return (fillColor.hasVboPtr() || fillColor.hasUboPtr() || fillColor.getUniformValue().a() > 0);
}

bool PolyMark::_doStroke() {
  return ((strokeWidth.hasVboPtr() || strokeWidth.hasUboPtr() || strokeWidth.getUniformValue() > 0) &&
          (strokeColor.hasVboPtr() || strokeColor.hasUboPtr() || strokeColor.getUniformValue().a() > 0));
}

void PolyMark::_updateShader() {
  if (!_shaderDirty || !_perGpuData.size()) {
    // early out
    return;
  }

  std::vector<std::string> polyShaders, lineShaders;
  bool doFill = _doFill();
  if (doFill) {
    polyShaders = {PolyTemplate_Vert::source, PolyTemplate_Frag::source};

    std::vector<BaseRenderProperty*> fillProps = {&x, &y, &fillColor};  // TODO(croot): add z
    _buildShaderSrc(polyShaders, fillProps, "PolyData", "polyData");
  }

  bool doStroke = _doStroke();
  if (doStroke) {
    lineShaders = {LineTemplate_Vert::source, LineTemplate_Frag::source, LineTemplate_Geom::source};

    std::vector<BaseRenderProperty*> strokeProps = {
        &x, &y, &strokeColor, &strokeWidth, &lineJoin, &miterLimit};  // TODO(croot): add z
    _buildShaderSrc(lineShaders, strokeProps, "LineData", "lineData");
  }

  // static int CROOTcnt = 0;
  // CROOTcnt++;
  // if (CROOTcnt == 1) {
  //   std::ofstream shadersrcstream;
  //   shadersrcstream.open("shadersource.vert");
  //   shadersrcstream << polyShaders[0];
  //   shadersrcstream.close();

  //   // shadersrcstream.open("shadersource.geom");
  //   // shadersrcstream << lineShaders[2];
  //   // shadersrcstream.close();

  //   shadersrcstream.open("shadersource.frag");
  //   shadersrcstream << polyShaders[1];
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

    if (doFill) {
      itr.second.shaderPtr = rsrcMgr->createShader(BaseMark::_addBaseTypeDefinesToShaderSrc(polyShaders[0]),
                                                   BaseMark::_addBaseTypeDefinesToShaderSrc(polyShaders[1]));
    }

    if (doStroke) {
      itr.second.strokeShaderPtr = rsrcMgr->createShader(BaseMark::_addBaseTypeDefinesToShaderSrc(lineShaders[0]),
                                                         BaseMark::_addBaseTypeDefinesToShaderSrc(lineShaders[1]),
                                                         BaseMark::_addBaseTypeDefinesToShaderSrc(lineShaders[2]));
    }

    // TODO(croot): should check the shader block layout attached
    // to the uniform buffer with that in the shader when the GLShader
    // is fixed to parse the shader code to add uniform interface block
    // attrs in the order they're defined in the shader because there
    // doesn't seem to be an opengl api to get those in order.

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

void PolyMark::_buildVAOData(const GpuId& gpuId,
                             ::Rendering::GL::Resources::GLShader* activeShader,
                             ::Rendering::GL::Resources::VboLayoutAttrToShaderAttrMap& attrMap,
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
                            ": Invalid poly mark. The sizes of the vertex buffer attributes do not match for gpuId " +
                            std::to_string(gpuId) + ". " + std::to_string(vboSize) + "!=" + std::to_string(itrSize));
    }

    itr->addToVboAttrMap(gpuId, attrMap);
  }

  CHECK(_dataPtr);
  QueryPolyDataTableShPtr polyTable = std::dynamic_pointer_cast<BaseQueryPolyDataTable>(_dataPtr);
  CHECK(polyTable);
  ibo = polyTable->getGLIndexBuffer(gpuId);
}

void PolyMark::_bindUniformProperties(::Rendering::GL::Resources::GLShader* activeShader,
                                      const std::set<BaseRenderProperty*>& props) {
  std::unordered_map<std::string, std::string> subroutines;

  for (auto prop : _vboProps) {
    if (props.find(prop) != props.end() && activeShader->hasVertexAttribute(prop->getName())) {
      const ScaleRefShPtr& scalePtr = prop->getScaleReference();
      if (scalePtr != nullptr) {
        scalePtr->bindUniformsToRenderer(activeShader, subroutines, "_" + prop->getName());
      }
    }
  }

  for (auto prop : _uboProps) {
    if (props.find(prop) != props.end()) {
      const ScaleRefShPtr& scalePtr = prop->getScaleReference();
      if (scalePtr != nullptr && activeShader->hasUniformBlockAttribute(prop->getDataColumnName())) {
        scalePtr->bindUniformsToRenderer(activeShader, subroutines, "_" + prop->getName());
      }
    }
  }

  for (auto prop : _uniformProps) {
    if (props.find(prop) != props.end() && activeShader->hasUniformAttribute(prop->getName())) {
      const ScaleRefShPtr& scalePtr = prop->getScaleReference();
      if (scalePtr != nullptr) {
        scalePtr->bindUniformsToRenderer(activeShader, subroutines, "_" + prop->getName());
      }

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

void PolyMark::_updateRenderPropertyGpuResources(const QueryRendererContext* ctx,
                                                 const std::set<GpuId>& usedGpus,
                                                 const std::set<GpuId>& unusedGpus) {
  x.initGpuResources(ctx, usedGpus, unusedGpus);
  y.initGpuResources(ctx, usedGpus, unusedGpus);
  // z.initGpuResources(ctx, usedGpus, unusedGpus);
  id.initGpuResources(ctx, usedGpus, unusedGpus);
  fillColor.initGpuResources(ctx, usedGpus, unusedGpus);
  strokeColor.initGpuResources(ctx, usedGpus, unusedGpus);
  strokeWidth.initGpuResources(ctx, usedGpus, unusedGpus);
  lineJoin.initGpuResources(ctx, usedGpus, unusedGpus);
  miterLimit.initGpuResources(ctx, usedGpus, unusedGpus);
}

void PolyMark::draw(::Rendering::GL::GLRenderer* renderer, const GpuId& gpuId) {
  // NOTE: shader should have been updated before calling this
  auto itr = _perGpuData.find(gpuId);
  if (itr == _perGpuData.end()) {
    return;
  }

  RootPerGpuDataShPtr qrmGpuData = itr->second.getRootPerGpuData();
  CHECK(qrmGpuData);

  ::Rendering::Renderer* rndr = qrmGpuData->rendererPtr.get();
  CHECK(rndr == renderer);

  CHECK(_dataPtr);
  QueryPolyDataTableShPtr polyTable = std::dynamic_pointer_cast<BaseQueryPolyDataTable>(_dataPtr);
  CHECK(polyTable);

  ::Rendering::GL::Resources::GLUniformBufferShPtr ubo;
  PolyRowDataShPtr rowDataPtr;

  // PropMap_by_Name& uboPropLookup = _uboProps.get<BaseMark::PropertyName>();

  if (itr->second.shaderPtr) {
    CHECK(itr->second.vaoPtr);

    // now bind the shader
    renderer->bindShader(itr->second.shaderPtr);

    // NOTE: the ibo will be bound with the vao object
    renderer->bindVertexArray(itr->second.vaoPtr);
    _bindUniformProperties(itr->second.shaderPtr.get(), _usedProps);

    if (_uboProps.size()) {
      // the same ubo should be used for all ubo props, so only need to grab
      // from first one
      ubo = (*_uboProps.begin())->getUboPtr(gpuId)->getGLUniformBufferPtr();
    }

    ::Rendering::GL::Resources::GLIndirectDrawIndexBufferShPtr indibo = polyTable->getGLIndirectDrawIndexBuffer(gpuId);
    CHECK(indibo && (!ubo || indibo->numItems() == ubo->numItems()));

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
  }

  // now draw outlines

  if (itr->second.strokeShaderPtr) {
    // reset the ubo to null and re-grab if there are stroke-related props in it
    ubo = nullptr;
    rowDataPtr = nullptr;

    CHECK(itr->second.strokeVaoPtr);

    // now bind the shader
    renderer->bindShader(itr->second.strokeShaderPtr);

    // set the viewport in the shader, the viewport is needed
    // to do screen-space line widths
    auto viewport = renderer->getViewport();
    itr->second.strokeShaderPtr->setUniformAttribute<int>("viewport.x", viewport.getXPos());
    itr->second.strokeShaderPtr->setUniformAttribute<int>("viewport.y", viewport.getYPos());
    itr->second.strokeShaderPtr->setUniformAttribute<int>("viewport.width", viewport.getWidth());
    itr->second.strokeShaderPtr->setUniformAttribute<int>("viewport.height", viewport.getHeight());

    // NOTE: the ibo will be bound with the vao object
    renderer->bindVertexArray(itr->second.strokeVaoPtr);
    _bindUniformProperties(itr->second.strokeShaderPtr.get(), _usedStrokeProps);

    if (_uboProps.size()) {
      // the same ubo should be used for all ubo props, so only need to grab
      // from first one
      ubo = (*_uboProps.begin())->getUboPtr(gpuId)->getGLUniformBufferPtr();

      renderer->bindUniformBuffer(ubo);

      rowDataPtr = polyTable->getRowDataPtr(gpuId);
    }

    ::Rendering::GL::Resources::GLIndirectDrawVertexBufferShPtr indvbo =
        polyTable->getGLIndirectDrawVertexBuffer(gpuId);

    CHECK(indvbo && (!ubo || (rowDataPtr && ubo->numItems() == rowDataPtr->size())));

    renderer->bindIndirectDrawBuffer(indvbo);

    if (ubo) {
      size_t startLineLoopIdx = 0;
      for (size_t i = 0; i < rowDataPtr->size(); ++i) {
        auto& rowData = (*rowDataPtr)[i];
        itr->second.strokeShaderPtr->bindUniformBufferToBlock("LineData", ubo, i);
        for (size_t j = 0; j < rowData.numLineLoops; ++j) {
          renderer->drawIndirectVertexBuffers(GL_LINE_STRIP_ADJACENCY, startLineLoopIdx + j, 1);
        }
        startLineLoopIdx += rowData.numLineLoops;
      }
    } else {
      renderer->drawIndirectVertexBuffers(GL_LINE_STRIP_ADJACENCY);
    }

    // reset state
    renderer->setLineWidth(1.0);
  }
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
