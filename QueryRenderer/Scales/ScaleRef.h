#ifndef QUERYRENDERER_SCALES_SCALEREF_H_
#define QUERYRENDERER_SCALES_SCALEREF_H_

#include "Types.h"
#include "../Types.h"
#include "../Marks/Types.h"
#include "Scale.h"
#include "ScaleDomainRangeData.h"

#include <Rendering/Renderer/GL/TypeGL.h>
#include <Rendering/Renderer/GL/Resources/GLShader.h>

// #include <QueryEngine/Execute.h>

#include <boost/algorithm/string/replace.hpp>

namespace QueryRenderer {

class BaseScaleRef {
 public:
  BaseScaleRef(const QueryRendererContextShPtr& ctx, const ScaleShPtr& scalePtr, BaseRenderProperty* rndrProp)
      : _ctx(ctx), _scalePtr(scalePtr), _rndrPropPtr(rndrProp) {}
  virtual ~BaseScaleRef() {}

  std::string getName() const;
  const std::string& getNameRef();

  virtual const ::Rendering::GL::TypeGLShPtr& getDomainTypeGL();

  virtual const ::Rendering::GL::TypeGLShPtr& getRangeTypeGL();

  std::string getScaleGLSLFuncName(const std::string& extraSuffix = "");

  virtual std::string getGLSLCode(const std::string& extraSuffix = "");

  virtual void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader,
                                      const std::string& extraSuffix = "");

  virtual void updateScaleRef(const ScaleShPtr& scalePtr) = 0;

  ScaleShPtr getScalePtr() { return _scalePtr; }

  virtual operator std::string() const = 0;

 protected:
  void _verifyScalePointer() const;

  std::string _printInfo() const;

  const QueryDataTableShPtr& _getDataTablePtr();
  std::string _getDataColumnName();
  std::string _getRndrPropName();

  QueryRendererContextShPtr _ctx;
  ScaleShPtr _scalePtr;

 private:
  BaseRenderProperty* _rndrPropPtr;
};

template <class T, class TT>
void convertDomainRangeData(std::unique_ptr<ScaleDomainRangeData<T>>& destData, ScaleDomainRangeData<TT>* srcData) {
  std::vector<TT>& srcVec = srcData->getVectorData();

  destData.reset(new ScaleDomainRangeData<T>(srcData->getName(), srcVec.size(), srcData->useString()));
  std::vector<T>& destVec = destData->getVectorData();
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
      : BaseScaleRef(ctx, scalePtr, rndrProp), _coercedDomainData(nullptr), _coercedRangeData(nullptr) {
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
      _scalePtr = scalePtr;
    }

    _updateDomainRange(_scalePtr->hasDomainDataChanged(), _scalePtr->hasRangeDataChanged());
  }

  void bindUniformsToRenderer(::Rendering::GL::Resources::GLShader* activeShader, const std::string& extraSuffix = "") {
    _verifyScalePointer();

    bool coerceDomain = (_coercedDomainData != nullptr);
    bool coerceRange = (_coercedRangeData != nullptr);

    if (coerceDomain) {
      activeShader->setUniformAttribute(_scalePtr->getDomainGLSLUniformName() + extraSuffix,
                                        _coercedDomainData->getVectorData());
    }

    if (coerceRange) {
      activeShader->setUniformAttribute(_scalePtr->getRangeGLSLUniformName() + extraSuffix,
                                        _coercedRangeData->getVectorData());
    }

    _scalePtr->bindUniformsToRenderer(activeShader, extraSuffix, coerceDomain, coerceRange);
  }

  operator std::string() const final {
    return "ScaleRef<" + std::string(typeid(DomainType).name()) + ", " + std::string(typeid(RangeType).name()) + "> " +
           _printInfo();
  }

 private:
  std::unique_ptr<ScaleDomainRangeData<DomainType>> _coercedDomainData;
  std::unique_ptr<ScaleDomainRangeData<RangeType>> _coercedRangeData;
  std::unique_ptr<RangeType> _coercedDefaultVal;

  void _updateDomainRange(bool updateDomain, bool updateRange, bool force = false) {
    CHECK(_scalePtr != nullptr);

    ScaleDomainRangeData<unsigned int>* uintDomain;
    ScaleDomainRangeData<int>* intDomain;
    ScaleDomainRangeData<float>* floatDomain;
    ScaleDomainRangeData<double>* doubleDomain;
    ScaleDomainRangeData<std::string>* stringDomain;
    ScaleDomainRangeData<::Rendering::Objects::ColorRGBA>* colorDomain;
    bool doSort = false;

    if (updateDomain) {
      BaseScaleDomainRangeData* domainDataPtr = _scalePtr->getDomainData();
      if (force || domainDataPtr->getTypeInfo() != typeid(DomainType)) {
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
    }

    if (updateRange) {
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
    }

    if (doSort) {
      _sort();
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

    SqlQueryDataTable* sqlDataTable = dynamic_cast<SqlQueryDataTable*>(vboDataTable.get());

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

    std::vector<std::string>& vec = domainData->getVectorData();
    _coercedDomainData.reset(
        new ScaleDomainRangeData<DomainType>(domainData->getName(), vec.size(), domainData->useString()));

    // std::vector<DomainType>& coercedVec = _coercedDomainData->getVectorData();
    // for (size_t i = 0; i < vec.size(); ++i) {
    //   // get data from the executor
    //   coercedVec[i] =
    //       static_cast<DomainType>(executor->getStringId(tableName, colName, vec[i], queryDataLayoutPtr.get()));
    // }
  }

  void _sort() {
    _verifyScalePointer();

    bool hasDomain = (_coercedDomainData != nullptr);
    bool hasRange = (_coercedRangeData != nullptr);

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

    std::vector<DomainType>& domainVec = _coercedDomainData->getVectorData();
    std::vector<RangeType>& rangeVec = _coercedRangeData->getVectorData();

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
  }
};

}  // namespace QueryRenderer

#endif  // QUERYRENDERER_SCALES_SCALEREF_H_
