#ifndef QUERYRENDERER_SCALES_SCALEREF_H_
#define QUERYRENDERER_SCALES_SCALEREF_H_

#include "Types.h"
#include "../Types.h"
#include "../Marks/Types.h"
#include "Scale.h"
#include "ScaleDomainRangeData.h"
#include "Utils.h"

#include <Rendering/Renderer/GL/TypeGL.h>
#include <Rendering/Renderer/GL/Resources/GLShader.h>

#include <QueryEngine/Execute.h>

#include <boost/algorithm/string/replace.hpp>

namespace QueryRenderer {

class BaseScaleRef {
 public:
  BaseScaleRef(const QueryRendererContextShPtr& ctx, const ScaleShPtr& scalePtr, BaseRenderProperty* rndrProp);

  virtual ~BaseScaleRef() { _deleteScalePtr(); }

  std::string getName() const;
  const std::string& getNameRef();

  virtual const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL();
  virtual const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL();

  std::string getDomainGLSLTypeName(const std::string& extraSuffix = "");
  std::string getRangeGLSLTypeName(const std::string& extraSuffix = "");
  std::string getScaleGLSLFuncName(const std::string& extraSuffix = "");

  const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL() const;
  const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL() const;

  virtual std::string getGLSLCode(const std::string& extraSuffix = "");

  virtual void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                      std::unordered_map<std::string, std::string>& subroutineMap,
                                      const std::string& extraSuffix = "");

  virtual BaseScaleDomainRangeData* getDomainData() = 0;
  virtual BaseScaleDomainRangeData* getRangeData() = 0;

  virtual void updateScaleRef(const ScaleShPtr& scalePtr) = 0;

  ScaleShPtr getScalePtr() { return _scalePtr; }

  AccumulatorType getAccumulatorType() const;
  bool hasAccumulator() const;

  virtual operator std::string() const = 0;

 protected:
  void _verifyScalePointer() const;

  std::string _printInfo() const;

  const QueryDataTableShPtr& _getDataTablePtr();
  std::string _getDataColumnName();
  std::string _getRndrPropName();

  void _initScalePtr(const ScaleDomainRangeDataShPtr& domainDataPtr, const ScaleDomainRangeDataShPtr& rangeDataPtr);
  void _deleteScalePtr();

  QueryRendererContextShPtr _ctx;
  ScaleShPtr _scalePtr;

 private:
  BaseRenderProperty* _rndrPropPtr;
};

template <class T, class TT, class Enable = void>
struct ConvertDomainRangeData {
  void operator()(const QueryRendererContextShPtr& ctx,
                  std::shared_ptr<ScaleDomainRangeData<T>>& destData,
                  ScaleDomainRangeData<TT>* srcData) {
    std::vector<TT>& srcVec = srcData->getVectorDataRef();

    destData.reset(new ScaleDomainRangeData<T>(ctx, srcData->getName(), srcVec.size(), srcData->useString()));
    std::vector<T>& destVec = destData->getVectorDataRef();
    for (size_t i = 0; i < srcVec.size(); ++i) {
      destVec[i] = static_cast<T>(srcVec[i]);
    }
  }
};

template <class T, class TT>
struct ConvertDomainRangeData<
    T,
    TT,
    typename std::enable_if<::Rendering::Colors::is_color<T>::value && std::is_arithmetic<TT>::value>::type> {
  void operator()(const QueryRendererContextShPtr& ctx,
                  std::shared_ptr<ScaleDomainRangeData<T>>& destData,
                  ScaleDomainRangeData<TT>* srcData) {
    THROW_RUNTIME_EX("Cannot convert a numeric value (" + std::string(*srcData) + ") to a color" +
                     (destData ? " (" + std::string(*destData) + ")." : "."));
  }
};

template <class T, class TT>
struct ConvertDomainRangeData<
    T,
    TT,
    typename std::enable_if<std::is_arithmetic<T>::value && Rendering::Colors::is_color<TT>::value>::type> {
  void operator()(const QueryRendererContextShPtr& ctx,
                  std::shared_ptr<ScaleDomainRangeData<T>>& destData,
                  ScaleDomainRangeData<TT>* srcData) {
    THROW_RUNTIME_EX("Cannot convert a color (" + std::string(*srcData) + ") to a numeric value" +
                     (destData ? " (" + std::string(*destData) + ")." : "."));
  }
};

template <class T, class TT>
struct ConvertDomainRangeData<
    T,
    TT,
    typename std::enable_if<::Rendering::Colors::is_color<T>::value && Rendering::Colors::is_color<TT>::value>::type> {
  void operator()(const QueryRendererContextShPtr& ctx,
                  std::shared_ptr<ScaleDomainRangeData<T>>& destData,
                  ScaleDomainRangeData<TT>* srcData) {
    std::vector<TT>& srcVec = srcData->getVectorDataRef();

    destData.reset(new ScaleDomainRangeData<T>(ctx, srcData->getName(), srcVec.size(), srcData->useString()));
    std::vector<T>& destVec = destData->getVectorDataRef();
    for (size_t i = 0; i < srcVec.size(); ++i) {
      convertColor(srcVec[i], destVec[i]);
    }
  }
};

template <typename DomainType, typename RangeType>
class ScaleRef : public BaseScaleRef {
 public:
  ScaleRef(const QueryRendererContextShPtr& ctx, const ScaleShPtr& scalePtr, BaseRenderProperty* rndrProp)
      : BaseScaleRef(ctx, scalePtr, rndrProp), _coercedDomainData(nullptr), _coercedRangeData(nullptr), _sorted(false) {
    _updateDomainRange(true, true);
    _initScalePtr(_coercedDomainData, _coercedRangeData);
  }

  ~ScaleRef() {}

  const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL() {
    _verifyScalePointer();
    if (_coercedDomainData) {
      return _coercedDomainData->getTypeGL();
    }

    return _scalePtr->getDomainTypeGL();
  }

  const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL() {
    _verifyScalePointer();
    if (_coercedRangeData) {
      return _coercedRangeData->getTypeGL();
    }

    return _scalePtr->getRangeTypeGL();
  }

  std::string getGLSLCode(const std::string& extraSuffix = "") {
    _verifyScalePointer();

    bool coerceDomain = (_coercedDomainData != nullptr);
    bool coerceRange = (_coercedRangeData != nullptr);

    // force to ignore the domain of the original scale
    std::string shaderCode = _scalePtr->getGLSLCode(extraSuffix, coerceDomain, coerceRange);

    if (coerceDomain) {
      boost::replace_first(shaderCode, "<domainType>", _coercedDomainData->getTypeGL()->glslType());
      boost::replace_all(shaderCode, "<domainTypeEnum>", std::to_string(_coercedDomainData->getTypeGL()->glslGLType()));
    }

    if (coerceRange) {
      boost::replace_first(shaderCode, "<rangeType>", _coercedRangeData->getTypeGL()->glslType());
      boost::replace_all(shaderCode, "<rangeTypeEnum>", std::to_string(_coercedRangeData->getTypeGL()->glslGLType()));
    }

    return shaderCode;
  }

  void updateScaleRef(const ScaleShPtr& scalePtr) {
    if (scalePtr != _scalePtr) {
      _deleteScalePtr();
      _scalePtr = scalePtr;
    }

    _updateDomainRange(_scalePtr->hasDomainDataChanged(), _scalePtr->hasRangeDataChanged());

    // if we get to this point, that means the scale is going to be used by a mark.
    // So we'll initialize any gpu resources the scale may use now.
    _initScalePtr(_coercedDomainData, _coercedRangeData);
  }

  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                              std::unordered_map<std::string, std::string>& subroutineMap,
                              const std::string& extraSuffix = "") {
    _verifyScalePointer();

    auto accumType = _scalePtr->getAccumulatorType();
    bool coerceDomain = (accumType != AccumulatorType::DENSITY && _coercedDomainData != nullptr);
    bool coerceRange = (accumType == AccumulatorType::UNDEFINED && _coercedRangeData != nullptr);

    if (coerceDomain) {
      activeShader->setUniformAttribute(_scalePtr->getDomainGLSLUniformName() + extraSuffix,
                                        _coercedDomainData->getVectorDataRef());

      auto domainUniformData = _scalePtr->getDomainTypeUniforms(extraSuffix, _coercedDomainData.get());
      auto& domainType = domainUniformData.first;
      auto& uniformMap = domainUniformData.second;

      for (auto& itr : uniformMap) {
        activeShader->setUniformAttribute(itr.first, convertType<DomainType>(domainType, itr.second));
      }
    }

    if (coerceRange) {
      activeShader->setUniformAttribute(_scalePtr->getRangeGLSLUniformName() + extraSuffix,
                                        _coercedRangeData->getVectorDataRef());

      auto rangeUniformData = _scalePtr->getRangeTypeUniforms(extraSuffix);
      auto& rangeType = rangeUniformData.first;
      auto& uniformMap = rangeUniformData.second;

      for (auto& itr : uniformMap) {
        activeShader->setUniformAttribute(itr.first, convertType<RangeType>(rangeType, itr.second));
      }
    }

    _scalePtr->bindUniformsToRenderer(activeShader, subroutineMap, extraSuffix, coerceDomain, coerceRange);
  }

  BaseScaleDomainRangeData* getDomainData() final {
    _verifyScalePointer();
    if (_coercedDomainData) {
      return _coercedDomainData.get();
    }

    return _scalePtr->getDomainData();
  }

  BaseScaleDomainRangeData* getRangeData() final {
    _verifyScalePointer();
    if (_coercedRangeData) {
      return _coercedRangeData.get();
    }

    return _scalePtr->getRangeData();
  }

  operator std::string() const final {
    return "ScaleRef<" + std::string(typeid(DomainType).name()) + ", " + std::string(typeid(RangeType).name()) + "> " +
           _printInfo();
  }

 private:
  std::shared_ptr<ScaleDomainRangeData<DomainType>> _coercedDomainData;
  std::shared_ptr<ScaleDomainRangeData<RangeType>> _coercedRangeData;
  bool _sorted;

  void _updateDomainRange(bool updateDomain, bool updateRange, bool force = false) {
    CHECK(_scalePtr != nullptr);

    ScaleDomainRangeData<unsigned int>* uintDomain;
    ScaleDomainRangeData<int>* intDomain;
    ScaleDomainRangeData<float>* floatDomain;
    ScaleDomainRangeData<uint64_t>* uint64Domain;
    ScaleDomainRangeData<int64_t>* int64Domain;
    ScaleDomainRangeData<double>* doubleDomain;
    ScaleDomainRangeData<std::string>* stringDomain;
    ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>* colorRGBADomain;
    ScaleDomainRangeData<::Rendering::Colors::ColorHSL>* colorHSLDomain;
    ScaleDomainRangeData<::Rendering::Colors::ColorLAB>* colorLABDomain;
    ScaleDomainRangeData<::Rendering::Colors::ColorHCL>* colorHCLDomain;

    bool prevSorted = _sorted, doSort = false, updatedDomain = false, updatedRange = false;

    // reset the sort flag
    _sorted = false;

    if (updateDomain) {
      BaseScaleDomainRangeData* domainDataPtr = _scalePtr->getDomainData(true);
      bool isQuantScale = isQuantitativeScale(_scalePtr->getType());
      const auto& theirDomainType = domainDataPtr->getTypeInfo();
      const auto& ourDomainType = typeid(DomainType);
      if (force || (!isQuantScale && theirDomainType != ourDomainType) ||
          (isQuantScale && !areTypesCompatible(theirDomainType, ourDomainType))) {
        if ((uintDomain = dynamic_cast<ScaleDomainRangeData<unsigned int>*>(domainDataPtr))) {
          ConvertDomainRangeData<DomainType, unsigned int>()(_ctx, _coercedDomainData, uintDomain);
        } else if ((intDomain = dynamic_cast<ScaleDomainRangeData<int>*>(domainDataPtr))) {
          ConvertDomainRangeData<DomainType, int>()(_ctx, _coercedDomainData, intDomain);
        } else if ((floatDomain = dynamic_cast<ScaleDomainRangeData<float>*>(domainDataPtr))) {
          ConvertDomainRangeData<DomainType, float>()(_ctx, _coercedDomainData, floatDomain);
        } else if ((uint64Domain = dynamic_cast<ScaleDomainRangeData<uint64_t>*>(domainDataPtr))) {
          ConvertDomainRangeData<DomainType, uint64_t>()(_ctx, _coercedDomainData, uint64Domain);
        } else if ((int64Domain = dynamic_cast<ScaleDomainRangeData<int64_t>*>(domainDataPtr))) {
          ConvertDomainRangeData<DomainType, int64_t>()(_ctx, _coercedDomainData, int64Domain);
        } else if ((doubleDomain = dynamic_cast<ScaleDomainRangeData<double>*>(domainDataPtr))) {
          ConvertDomainRangeData<DomainType, double>()(_ctx, _coercedDomainData, doubleDomain);
        } else if ((stringDomain = dynamic_cast<ScaleDomainRangeData<std::string>*>(domainDataPtr))) {
          _doStringToDataConversion(stringDomain);
          doSort = true;
        } else {
          THROW_RUNTIME_EX(std::string(*this) + ": Cannot create scale reference - unsupported domain type.");
        }
      } else {
        _coercedDomainData = nullptr;
      }

      updatedDomain = true;
    }

    // make sure to un-sort the range if it was previously sorted
    if (updateRange || (!doSort && prevSorted)) {
      BaseScaleDomainRangeData* rangeDataPtr = _scalePtr->getRangeData(true);
      if (force || rangeDataPtr->getTypeInfo() != typeid(RangeType)) {
        if ((uintDomain = dynamic_cast<ScaleDomainRangeData<unsigned int>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, unsigned int>()(_ctx, _coercedRangeData, uintDomain);
        } else if ((intDomain = dynamic_cast<ScaleDomainRangeData<int>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, int>()(_ctx, _coercedRangeData, intDomain);
        } else if ((floatDomain = dynamic_cast<ScaleDomainRangeData<float>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, float>()(_ctx, _coercedRangeData, floatDomain);
        } else if ((uint64Domain = dynamic_cast<ScaleDomainRangeData<uint64_t>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, uint64_t>()(_ctx, _coercedRangeData, uint64Domain);
        } else if ((int64Domain = dynamic_cast<ScaleDomainRangeData<int64_t>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, int64_t>()(_ctx, _coercedRangeData, int64Domain);
        } else if ((doubleDomain = dynamic_cast<ScaleDomainRangeData<double>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, double>()(_ctx, _coercedRangeData, doubleDomain);
        }
        // TODO(croot): support other strings?
        // else if ((stringDomain = dynamic_cast<ScaleDomainRangeData<std::string>*>(rangeDataPtr))) {
        // }
        else if ((colorRGBADomain =
                      dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorRGBA>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, ::Rendering::Colors::ColorRGBA>()(_ctx, _coercedRangeData, colorRGBADomain);
        } else if ((colorHSLDomain =
                        dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHSL>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, ::Rendering::Colors::ColorHSL>()(_ctx, _coercedRangeData, colorHSLDomain);
        } else if ((colorLABDomain =
                        dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorLAB>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, ::Rendering::Colors::ColorLAB>()(_ctx, _coercedRangeData, colorLABDomain);
        } else if ((colorHCLDomain =
                        dynamic_cast<ScaleDomainRangeData<::Rendering::Colors::ColorHCL>*>(rangeDataPtr))) {
          ConvertDomainRangeData<RangeType, ::Rendering::Colors::ColorHCL>()(_ctx, _coercedRangeData, colorHCLDomain);
        } else {
          THROW_RUNTIME_EX(std::string(*this) + ": Cannot create scale reference - unsupported range type.");
        }
      } else {
        _coercedRangeData = nullptr;
      }

      updatedRange = true;
    }

    if (!force && (doSort || (prevSorted && !updateDomain && updateRange))) {
      // make sure to redo the sort if the domain wasn't updated, but the range was
      // and the data was previously sorted
      _sort(updatedDomain, updatedRange);
    }
  }

  void _doStringToDataConversion(ScaleDomainRangeData<std::string>* domainData) {
    const Executor* const executor = _ctx->getExecutor();

    RUNTIME_EX_ASSERT(executor != nullptr,
                      std::string(*this) + ": An executor is not defined. Cannot numerically convert a string column.");

    const QueryDataTableShPtr& dataTable = _getDataTablePtr();

    RUNTIME_EX_ASSERT(dataTable != nullptr,
                      std::string(*this) + ": A data table is not referenced by render property \"" +
                          _getRndrPropName() + "\". Cannot numerically convert a string column.");

    RUNTIME_EX_ASSERT(dataTable->getBaseType() == QueryDataTableBaseType::BASIC_VBO,
                      std::string(*this) + ": data table referenced is not the correct type. It is " +
                          to_string(dataTable->getBaseType()) + " but needs to be a vertex buffer table type.");

    QueryDataTableVBOShPtr vboDataTable = std::dynamic_pointer_cast<BaseQueryDataTableVBO>(dataTable);
    CHECK(vboDataTable != nullptr);

    SqlQueryDataTableJSON* sqlDataTable = dynamic_cast<SqlQueryDataTableJSON*>(vboDataTable.get());

    RUNTIME_EX_ASSERT(sqlDataTable != nullptr,
                      std::string(*this) + ": The data table referenced by render property \"" + _getRndrPropName() +
                          "\" is not an sql data table. Cannot numerically convert a string column");

    std::string tableName = sqlDataTable->getTableName();
    RUNTIME_EX_ASSERT(
        tableName.length() != 0,
        std::string(*this) + "The sql data table referenced by render property \"" + _getRndrPropName() +
            "\" is not properly initialized. It is missing a \"dbTableName\" property. Cannot numerically "
            "convert a string column");

    std::string colName = _getDataColumnName();
    RUNTIME_EX_ASSERT(
        colName.length() != 0,
        std::string(*this) + ": The render property \"" + _getRndrPropName() +
            "\" is missing a column name to reference in the data. Cannot numerically convert a string column.");

    std::vector<std::string>& vec = domainData->getVectorDataRef();
    _coercedDomainData.reset(
        new ScaleDomainRangeData<DomainType>(_ctx, domainData->getName(), vec.size(), domainData->useString()));

    QueryDataLayoutShPtr queryDataLayoutPtr = sqlDataTable->getVboQueryDataLayout();
    // TODO(croot): check whether the colName exists in the layout
    if (!queryDataLayoutPtr || !queryDataLayoutPtr->hasAttribute(colName)) {
      queryDataLayoutPtr = sqlDataTable->getUboQueryDataLayout();
      if (!queryDataLayoutPtr) {
        // We can run into a scenario where there still exists a referenced data ptr, but that
        // data ptr has nothing in it. We handle that case by just returning here, but this
        // scale ref should/will not do anything in the future (like init a shader or bind uniforms)
        // We can reach here and ultimately not render thanks to a scale update in QueryRenderer::_initFromJSON
        return;
      }

      RUNTIME_EX_ASSERT(queryDataLayoutPtr->hasAttribute(colName),
                        "The vega data " + std::string(*sqlDataTable) +
                            " does not have a layout that contains the attribute \"" + colName + "\"");
    }

    std::vector<DomainType>& coercedVec = _coercedDomainData->getVectorDataRef();
    for (size_t i = 0; i < vec.size(); ++i) {
      // get data from the executor
      coercedVec[i] = static_cast<DomainType>(
          executor->getStringId(tableName, colName, vec[i], queryDataLayoutPtr.get(), sqlDataTable->getResultRows()));
    }
  }

  void _sort(bool domainUpdated, bool rangeUpdated) {
    _verifyScalePointer();

    bool hasDomain = (domainUpdated && _coercedDomainData != nullptr);
    bool hasRange = (rangeUpdated && _coercedRangeData != nullptr);

    // force a copy of both the domain and range to sort
    _updateDomainRange(!hasDomain, !hasRange, true);

    // TODO(croot): somehow do a sort in place? Not sure how to do this without
    // creating an iterator class on the ScaleRef objects (which might be nice
    // to do in the future). So, for now, I'm just copying all the domain/range
    // data as pairs into a vector, sorting that vector based on the domain, and
    // placing the results back. Very hacky, but since domains/ranges should on the
    // whole be small, this shouldn't be a big bottle neck.

    // TODO(croot): Possible bug -- the size of the domains/ranges don't have to
    // be equal. You can have more domains than ranges and vice-versa. So we need
    // to sort by the smaller of the two and leave the hanging items alone.

    std::vector<DomainType>& domainVec = _coercedDomainData->getVectorDataRef();
    std::vector<RangeType>& rangeVec = _coercedRangeData->getVectorDataRef();

    int numItems = std::min(domainVec.size(), rangeVec.size());
    std::vector<std::pair<DomainType, RangeType>> sortVec(numItems);

    int i;
    for (i = 0; i < numItems; ++i) {
      sortVec[i] = std::make_pair(domainVec[i], rangeVec[i]);
    }

    std::sort(sortVec.begin(),
              sortVec.end(),
              [](const std::pair<DomainType, RangeType>& a, const std::pair<DomainType, RangeType>& b) {
                return a.first < b.first;
              });

    for (i = 0; i < numItems; ++i) {
      domainVec[i] = sortVec[i].first;
      rangeVec[i] = sortVec[i].second;
    }

    _sorted = true;
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALEREF_H_
