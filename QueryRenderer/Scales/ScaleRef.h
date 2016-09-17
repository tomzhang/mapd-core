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
  void _initScalePtr();
  void _deleteScalePtr();

  QueryRendererContextShPtr _ctx;
  ScaleShPtr _scalePtr;

 private:
  BaseRenderProperty* _rndrPropPtr;
};

template <class T, class TT>
void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<T>>& destData, ScaleDomainRangeData<TT>* srcData) {
  std::vector<TT>& srcVec = srcData->getVectorDataRef();

  destData.reset(new ScaleDomainRangeData<T>(srcData->getName(), srcVec.size(), srcData->useString()));
  std::vector<T>& destVec = destData->getVectorDataRef();
  for (size_t i = 0; i < srcVec.size(); ++i) {
    destVec[i] = static_cast<T>(srcVec[i]);
  }
}

template <class T>
void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<T>>& destData,
                            ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>* srcData) {
  THROW_RUNTIME_EX("Cannot convert a color (" + std::string(*srcData) + ") to a numeric value (" +
                   std::string(*destData) + ").");
}

template <class TT>
void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>>& destData,
                            ScaleDomainRangeData<TT>* srcData) {
  THROW_RUNTIME_EX("Cannot convert a numeric value (" + std::string(*srcData) + ") to a color (" +
                   std::string(*destData) + ").");
}

void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>>& destData,
                            ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>* srcData);

template <typename DomainType, typename RangeType>
class ScaleRef : public BaseScaleRef {
 public:
  ScaleRef(const QueryRendererContextShPtr& ctx, const ScaleShPtr& scalePtr, BaseRenderProperty* rndrProp)
      : BaseScaleRef(ctx, scalePtr, rndrProp), _coercedDomainData(nullptr), _coercedRangeData(nullptr), _sorted(false) {
    _updateDomainRange(true, true);
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
    }

    if (coerceRange) {
      boost::replace_first(shaderCode, "<rangeType>", _coercedRangeData->getTypeGL()->glslType());
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
    _initScalePtr();
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

  operator std::string() const final {
    return "ScaleRef<" + std::string(typeid(DomainType).name()) + ", " + std::string(typeid(RangeType).name()) + "> " +
           _printInfo();
  }

 private:
  std::unique_ptr<ScaleDomainRangeData<DomainType>> _coercedDomainData;
  std::unique_ptr<ScaleDomainRangeData<RangeType>> _coercedRangeData;
  bool _sorted;

  void _updateDomainRange(bool updateDomain, bool updateRange, bool force = false) {
    CHECK(_scalePtr != nullptr);

    ScaleDomainRangeData<unsigned int>* uintDomain;
    ScaleDomainRangeData<int>* intDomain;
    ScaleDomainRangeData<float>* floatDomain;
    ScaleDomainRangeData<double>* doubleDomain;
    ScaleDomainRangeData<std::string>* stringDomain;
    ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>* colorDomain;

    bool prevSorted = _sorted, doSort = false, updatedDomain = false, updatedRange = false;

    // reset the sort flag
    _sorted = false;

    if (updateDomain) {
      BaseScaleDomainRangeData* domainDataPtr = _scalePtr->getDomainData();
      bool isQuantScale = isQuantitativeScale(_scalePtr->getType());
      const auto& theirDomainType = domainDataPtr->getTypeInfo();
      const auto& ourDomainType = typeid(DomainType);
      if (force || (!isQuantScale && theirDomainType != ourDomainType) ||
          (isQuantScale && !areTypesCompatible(ourDomainType, theirDomainType))) {
        uintDomain = dynamic_cast<ScaleDomainRangeData<unsigned int>*>(domainDataPtr);
        intDomain = dynamic_cast<ScaleDomainRangeData<int>*>(domainDataPtr);
        floatDomain = dynamic_cast<ScaleDomainRangeData<float>*>(domainDataPtr);
        doubleDomain = dynamic_cast<ScaleDomainRangeData<double>*>(domainDataPtr);
        stringDomain = dynamic_cast<ScaleDomainRangeData<std::string>*>(domainDataPtr);

        if (uintDomain) {
          convertDomainRangeData(_coercedDomainData, uintDomain);
        } else if (intDomain) {
          convertDomainRangeData(_coercedDomainData, intDomain);
        } else if (floatDomain) {
          convertDomainRangeData(_coercedDomainData, floatDomain);
        } else if (doubleDomain) {
          convertDomainRangeData(_coercedDomainData, doubleDomain);
        } else if (stringDomain) {
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
      BaseScaleDomainRangeData* rangeDataPtr = _scalePtr->getRangeData();
      if (force || rangeDataPtr->getTypeInfo() != typeid(RangeType)) {
        uintDomain = dynamic_cast<ScaleDomainRangeData<unsigned int>*>(rangeDataPtr);
        intDomain = dynamic_cast<ScaleDomainRangeData<int>*>(rangeDataPtr);
        floatDomain = dynamic_cast<ScaleDomainRangeData<float>*>(rangeDataPtr);
        doubleDomain = dynamic_cast<ScaleDomainRangeData<double>*>(rangeDataPtr);
        stringDomain = dynamic_cast<ScaleDomainRangeData<std::string>*>(rangeDataPtr);
        colorDomain = dynamic_cast<ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>*>(rangeDataPtr);

        if (uintDomain) {
          convertDomainRangeData(_coercedRangeData, uintDomain);
        } else if (intDomain) {
          convertDomainRangeData(_coercedRangeData, intDomain);
        } else if (floatDomain) {
          convertDomainRangeData(_coercedRangeData, floatDomain);
        } else if (doubleDomain) {
          convertDomainRangeData(_coercedRangeData, doubleDomain);
        } else if (force && colorDomain) {
          convertDomainRangeData(_coercedRangeData, colorDomain);
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

    QueryDataLayoutShPtr queryDataLayoutPtr = sqlDataTable->getQueryDataLayout();
    CHECK(queryDataLayoutPtr != nullptr);

    std::vector<std::string>& vec = domainData->getVectorDataRef();
    _coercedDomainData.reset(
        new ScaleDomainRangeData<DomainType>(domainData->getName(), vec.size(), domainData->useString()));

    std::vector<DomainType>& coercedVec = _coercedDomainData->getVectorDataRef();
    for (size_t i = 0; i < vec.size(); ++i) {
      // get data from the executor
      coercedVec[i] =
          static_cast<DomainType>(executor->getStringId(tableName, colName, vec[i], queryDataLayoutPtr.get()));
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
              [](const std::pair<DomainType, RangeType>& a,
                 const std::pair<DomainType, RangeType>& b) { return a.first < b.first; });

    for (i = 0; i < numItems; ++i) {
      domainVec[i] = sortVec[i].first;
      rangeVec[i] = sortVec[i].second;
    }

    _sorted = true;
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALEREF_H_
