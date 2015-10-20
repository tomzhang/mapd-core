#include "QueryRendererObjects.h"
#include "ShaderUtils.h"
#include <iostream>
#include <sstream>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/find.hpp>


#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/algorithm/string/find_iterator.hpp>
#include <boost/algorithm/string/regex_find_format.hpp>


using namespace MapD_Renderer;





typedef std::string::iterator str_itr;
typedef boost::iterator_range<str_itr> str_itr_range;

str_itr_range getGLSLFunctionBounds(std::string& codeStr, const std::string& funcName) {
    std::string regexStr = "\\h*\\w+\\h+" + funcName + "\\h*\\([\\w\\h\\v,]*\\)\\h*\\v*\\{";
    boost::regex funcSignatureRegex(regexStr);

    str_itr_range signature_range = boost::find_regex(codeStr, funcSignatureRegex);

    if (signature_range.empty()) {
        return signature_range;
    }

    str_itr lastItr = signature_range.end();
    std::vector<str_itr> scopeStack = { lastItr - 1 };
    // std::vector<str_itr> scopeStack;
    // scopeStack.push_back(signature_range.end());

    size_t curr_pos = signature_range.end() - codeStr.begin();
    while ((curr_pos = codeStr.find_first_of("{}", curr_pos)) != std::string::npos) {
        if (codeStr[curr_pos] == '{') {
            scopeStack.push_back(codeStr.begin() + curr_pos);
        } else {
            // found a '}'
            scopeStack.pop_back();
            if (scopeStack.empty()) {
                lastItr = codeStr.begin() + curr_pos + 1;
                break;
            }
        }
    }


    if (!scopeStack.empty()) {
        // return an empty range
        return str_itr_range();
    }

    return str_itr_range(signature_range.begin(), lastItr);



    // boost::regex bracketRegex("[\\{\\}]");
    // for (boost::find_iterator<str_itr> it = boost::make_find_iterator(codeStr.begin(), regex_finder(bracketRegex));
    //      it != boost::find_iterator<str_itr>(); ++it) {
    //     str_itr sitr = (*it).begin();
    //     if (*sitr == '{') {
    //         scopeStack.push_back(sitr);
    //     } else {
    //         scopeStack.pop_back();

    //         if (scopeStack.empty()) {
    //             break;
    //         }
    //     }
    // }



    // for (std::string::iterator it = signature_range.begin(); it != signature_range.end(); ++it) {
    //     std::cout << *it;
    // }
    // std::cout << std::endl;
}








DataType getDataTypeFromJSONObj(const rapidjson::Value& obj) {
    DataType rtn;
    rapidjson::Type type = obj.GetType();

    switch(type) {
        case rapidjson::kNumberType:
            if (obj.IsInt()) {
                rtn = DataType::INT;
            } else if (obj.IsUint()) {
                rtn = DataType::UINT;
            } else if (obj.IsDouble()) {
                double val = obj.GetDouble();
                if (val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::lowest()) {
                    rtn = DataType::FLOAT;
                } else {
                    rtn = DataType::DOUBLE;
                }
            } else {
                assert(false);
            }
            break;
        case rapidjson::kStringType:
            {
                std::string val = obj.GetString();
                if (ColorRGBA::isColorString(val)) {
                    rtn = DataType::COLOR;
                }
                else {
                    // TODO: throw exception instead of assert
                    assert(false);
                }
            }
            break;
        default:
            assert(false);
    }

    return rtn;
}



DataType getDataTypeFromDataRefJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
    assert(obj.IsObject());

    rapidjson::Value::ConstMemberIterator mitr;
    assert((mitr = obj.FindMember("data")) != obj.MemberEnd() && mitr->value.IsString());
    DataTableShPtr tablePtr = ctx->getDataTable(mitr->value.GetString());

    assert(tablePtr != nullptr);

    assert((mitr = obj.FindMember("field")) != obj.MemberEnd() && mitr->value.IsString());
    return tablePtr->getColumnType(mitr->value.GetString());
}





const std::vector<std::string> BaseScale::scaleVertexShaderFilenames = {
    "shaders/linearScaleTemplate.vert",    // LINEAR
    "shaders/ordinalScaleTemplate.vert"    // ORDINAL
};

BaseScale::BaseScale(const QueryRendererContextShPtr& ctx) : name(""), type(ScaleType::LINEAR), _domainType(nullptr), _rangeType(nullptr), _ctx(ctx), _useClamp(false) {}

BaseScale::BaseScale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(ctx) {
    _initFromJSONObj(obj);
}

BaseScale::~BaseScale() {
    // std::cerr << "IN BaseScale DESTRUCTOR " << name << std::endl;
}

void BaseScale::_initFromJSONObj(const rapidjson::Value& obj) {
    // TODO: throw exceptions instead of asserts
    assert(obj.IsObject());

    rapidjson::Value::ConstMemberIterator itr;
    assert((itr = obj.FindMember("name")) != obj.MemberEnd() &&
           itr->value.IsString());

    name = itr->value.GetString();

    assert((itr = obj.FindMember("type")) != obj.MemberEnd() &&
           itr->value.IsString());
    std::string strScaleType(itr->value.GetString());

    if (strScaleType == "linear") {
        type = ScaleType::LINEAR;

        if ((itr = obj.FindMember("clamp")) != obj.MemberEnd()) {
            assert(itr->value.IsBool());

            _useClamp = itr->value.GetBool();
        }
    } else if (strScaleType == "ordinal") {
        type = ScaleType::ORDINAL;
    } else {
        assert(false);
    }
}


std::string BaseScale::getScaleGLSLFuncName() {
    std::string scaleName;

    switch(type) {
        case ScaleType::LINEAR:
            scaleName = "Linear";
            break;
        case ScaleType::ORDINAL:
            scaleName = "Ordinal";
            break;
        default:
            assert(false);
    }

    std::ostringstream ss;

    ss << "get" << scaleName << "Scale_" << name;

    return ss.str();
}








template <typename T>
void ScaleDomainRangeData<T>::initializeFromJSONObj(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx, BaseScale::ScaleType type) {
    rapidjson::Value::ConstMemberIterator mitr;
    rapidjson::Value::ConstValueIterator vitr;

    bool isObject, isString = false;
    DataTableShPtr tablePtr;
    DataColumnShPtr columnPtr;

    if (_useString) {
        assert((mitr = obj.FindMember(_name.c_str())) != obj.MemberEnd() &&
           ((isObject = mitr->value.IsObject()) || (isString = mitr->value.IsString()) ||
            (mitr->value.IsArray() && mitr->value.Size())));
    } else {
        assert((mitr = obj.FindMember(_name.c_str())) != obj.MemberEnd() &&
               ((isObject = mitr->value.IsObject()) || (mitr->value.IsArray() && mitr->value.Size())));
    }

    const rapidjson::Value& jsonObj = mitr->value;

    if (isObject) {
        assert((mitr = jsonObj.FindMember("data")) != jsonObj.MemberEnd() && mitr->value.IsString());
        tablePtr = ctx->getDataTable(mitr->value.GetString());

        assert((mitr = jsonObj.FindMember("field")) != jsonObj.MemberEnd() && mitr->value.IsString());
        columnPtr = tablePtr->getColumn(mitr->value.GetString());

        TDataColumn<T> *dataColumnPtr = dynamic_cast<TDataColumn<T> *>(columnPtr.get());

        _vectorPtr = dataColumnPtr->getColumnData();

        _updateVectorDataByType(dataColumnPtr, type);
    } else if (isString) {
        _setFromStringValue(mitr->value.GetString(), type);
    } else {
        _vectorPtr.reset(new std::vector<T>());

        // gather all the items
        DataType itemType;
        for (vitr = jsonObj.Begin(); vitr != jsonObj.End(); ++vitr) {
            itemType = getDataTypeFromJSONObj(*vitr);

            // TODO: Throw an exception
            assert(itemType == dataType);

            _pushItem(*vitr);
        }
    }
}


template <typename T>
inline TypeGLShPtr ScaleDomainRangeData<T>::getTypeGL() {
    return TypeGLShPtr(new TypeGL<T>());
}

template <>
inline TypeGLShPtr ScaleDomainRangeData<ColorRGBA>::getTypeGL() {
    // TODO: combine all the different types into a utility file somewhere.
    // i.e. this is already defined in BufferLayout.h, so let's find a
    // good way to consolidate these definitions
    return TypeGLShPtr(new TypeGL<float, 4>());
}


template <>
const DataType ScaleDomainRangeData<unsigned int>::dataType = DataType::UINT;

template <>
void ScaleDomainRangeData<unsigned int>::_pushItem(const rapidjson::Value& obj) {
    _vectorPtr->push_back(obj.GetUint());
}

template <>
const DataType ScaleDomainRangeData<int>::dataType = DataType::INT;

template <>
void ScaleDomainRangeData<int>::_pushItem(const rapidjson::Value& obj) {
    _vectorPtr->push_back(obj.GetInt());
}

template <>
const DataType ScaleDomainRangeData<float>::dataType = DataType::FLOAT;

template <>
void ScaleDomainRangeData<float>::_pushItem(const rapidjson::Value& obj) {
    _vectorPtr->push_back(static_cast<float>(obj.GetDouble()));
}

template <>
const DataType ScaleDomainRangeData<double>::dataType = DataType::DOUBLE;

template <>
void ScaleDomainRangeData<double>::_pushItem(const rapidjson::Value& obj) {
    _vectorPtr->push_back(obj.GetDouble());
}

template <>
const DataType ScaleDomainRangeData<ColorRGBA>::dataType = DataType::COLOR;

template <>
void ScaleDomainRangeData<ColorRGBA>::_pushItem(const rapidjson::Value& obj) {
    ColorRGBA color(obj.GetString());
    _vectorPtr->push_back(color);
}



template <typename T>
void ScaleDomainRangeData<T>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type) {
    if (type == BaseScale::ScaleType::LINEAR && (strVal == "width" || strVal == "height")) {
        // TODO: Handle actual width/height vals
        _vectorPtr.reset(new std::vector<T>({0, 1}));

        // need to put the widht/height range from -1 to 1. Didn't do that
        // in the previous line because the compiler was complaining that
        // you cannot convert -1 to an unsigned int, tho if range is
        // an unsigned int, you should never get here. Doing a subtraction here
        // to get a -1 to 1 range without the compilation error.

        (*_vectorPtr)[0] -= 1;
    } else {
        // TODO: throw an exception
        assert(false);
    }

}

template <>
void ScaleDomainRangeData<ColorRGBA>::_setFromStringValue(const std::string& strVal, BaseScale::ScaleType type) {
    // TODO: throw an exception
    assert(false);
}




template <typename T>
void ScaleDomainRangeData<T>::_updateVectorDataByType(TDataColumn<T> *dataColumnPtr, BaseScale::ScaleType type) {
    if (type == BaseScale::ScaleType::LINEAR) {
        std::pair<T, T> minmaxDomain = dataColumnPtr->getExtrema();

        _vectorPtr.reset(new std::vector<T>({minmaxDomain.first, minmaxDomain.second}));
    }
}

template <>
void ScaleDomainRangeData<ColorRGBA>::_updateVectorDataByType(TDataColumn<ColorRGBA> *dataColumnPtr, BaseScale::ScaleType type) {
    if (type == BaseScale::ScaleType::LINEAR) {
        // TODO: throw an exception, or update ColorRGBA to be supported by the <,>,etc. operators
        // so that the getExtrema() call will work to get the min and max colors
        assert(false);
    }
}






template <typename DomainType, typename RangeType>
Scale<DomainType, RangeType>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
    _initGLTypes();
    _initFromJSONObj(obj);
}




// template <>
// Scale<unsigned int, unsigned int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::UINT, DataType::UINT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<unsigned int, int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::UINT, DataType::INT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<unsigned int, float>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::UINT, DataType::FLOAT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<unsigned int, double>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::UINT, DataType::DOUBLE, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<unsigned int, ColorRGBA>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::UINT, DataType::COLOR, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }


// template <>
// Scale<int, unsigned int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::INT, DataType::UINT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<int, int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::INT, DataType::INT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<int, float>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::INT, DataType::FLOAT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<int, double>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::INT, DataType::DOUBLE, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<int, ColorRGBA>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::INT, DataType::COLOR, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }


// template <>
// Scale<float, unsigned int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::FLOAT, DataType::UINT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<float, int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::FLOAT, DataType::INT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<float, float>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::FLOAT, DataType::FLOAT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<float, double>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::FLOAT, DataType::DOUBLE, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<float, ColorRGBA>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::FLOAT, DataType::COLOR, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }



// template <>
// Scale<double, unsigned int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::DOUBLE, DataType::UINT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<double, int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::DOUBLE, DataType::INT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<double, float>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::DOUBLE, DataType::FLOAT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<double, double>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::DOUBLE, DataType::DOUBLE, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<double, ColorRGBA>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::DOUBLE, DataType::COLOR, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }




// template <>
// Scale<ColorRGBA, unsigned int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::COLOR, DataType::UINT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<ColorRGBA, int>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::COLOR, DataType::INT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<ColorRGBA, float>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::COLOR, DataType::FLOAT, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<ColorRGBA, double>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::COLOR, DataType::DOUBLE, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }

// template <>
// Scale<ColorRGBA, ColorRGBA>::Scale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) : BaseScale(DataType::COLOR, DataType::COLOR, obj, ctx), _domainPtr("domain", false), _rangePtr("range", true) {
//     _initGLTypes();
//     _initFromJSONObj(obj);
// }




template <typename DomainType, typename RangeType>
Scale<DomainType, RangeType>::~Scale() {
    // std::cerr << "IN Scale DESTRUCTOR " << name << std::endl;
}



// TODO: use template specialization instead of
// a switch inside these functions

// template <typename DomainType, typename RangeType>
// void Scale<DomainType, RangeType>::_pushDomainItem(const rapidjson::Value& item) {
//     DataType type = getDataTypeFromJSONObj(item);
//     assert(type == domainType);

//     // _domainPtr.pushItem(item);

//     // switch(type) {
//     //     case DataType::UINT:
//     //         _domainPtr->push_back(item.GetUint());
//     //         break;
//     //     case DataType::INT:
//     //         _domainPtr->push_back(item.GetInt());
//     //         break;
//     //     case DataType::FLOAT:
//     //         _domainPtr->push_back(static_cast<float>(item.GetDouble()));
//     //         break;
//     //     case DataType::DOUBLE:
//     //         _domainPtr->push_back(item.GetDouble());
//     //         break;
//     //     // case DataType::COLOR:
//     //     //     _domainPtr->push_back(ColorRGBA(item.GetString()));
//     //     //     break;
//     //     default:
//     //         assert(false);
//     //         break;
//     // }
// }

// template <typename DomainType, typename RangeType>
// void Scale<DomainType, RangeType>::_pushRangeItem(const rapidjson::Value& item) {
//     DataType type = getDataTypeFromJSONObj(item);
//     assert(type == rangeType);

//     // _rangePtr.pushItem(item);

//     // switch(type) {
//     //     case DataType::UINT:
//     //         _rangePtr->push_back(item.GetUint());
//     //         break;
//     //     case DataType::INT:
//     //         _rangePtr->push_back(item.GetInt());
//     //         break;
//     //     case DataType::FLOAT:
//     //         _rangePtr->push_back(static_cast<float>(item.GetDouble()));
//     //         break;
//     //     case DataType::DOUBLE:
//     //         _rangePtr->push_back(item.GetDouble());
//     //         break;
//     //     case DataType::COLOR:
//     //         {
//     //             ColorRGBA color(item.GetString());
//     //             _rangePtr->push_back(color);
//     //         }
//     //         break;
//     //     default:
//     //         assert(false);
//     //         break;
//     // }
// }


template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::_initFromJSONObj(const rapidjson::Value& obj) {

    _domainPtr.initializeFromJSONObj(obj, _ctx, type);
    _rangePtr.initializeFromJSONObj(obj, _ctx, type);


    // rapidjson::Value::ConstMemberIterator mitr;
    // rapidjson::Value::ConstValueIterator vitr;

    // bool isObject, isString;
    // DataTableShPtr tablePtr;
    // DataColumnShPtr columnPtr;

    // assert((mitr = obj.FindMember("domain")) != obj.MemberEnd() &&
    //        ((isObject = mitr->value.IsObject()) || (mitr->value.IsArray() && mitr->value.Size())));

    // const rapidjson::Value& domainObj = mitr->value;

    // if (isObject) {
    //     assert((mitr = domainObj.FindMember("data")) != domainObj.MemberEnd() && mitr->value.IsString());
    //     tablePtr = _ctx->getDataTable(mitr->value.GetString());

    //     assert((mitr = domainObj.FindMember("field")) != domainObj.MemberEnd() && mitr->value.IsString());
    //     columnPtr = tablePtr->getColumn(mitr->value.GetString());

    //     TDataColumn<DomainType> *domainColumnPtr = dynamic_cast<TDataColumn<DomainType> *>(columnPtr.get());

    //     // _domainPtr = domainColumnPtr->getColumnData();
    //     _domainPtr.setVectorPtr(domainColumnPtr->getColumnData());

    //     if (type == ScaleType::LINEAR) {
    //         std::pair<DomainType, DomainType> minmaxDomain = domainColumnPtr->getExtrema();

    //         // _domainPtr.reset(new std::vector<DomainType>({minmaxDomain.first, minmaxDomain.second}));
    //         _domainPtr.initializeNewVectorPtr({minmaxDomain.first, minmaxDomain.second});
    //     }
    // } else {
    //     // _domainPtr.reset(new std::vector<DomainType>());
    //     _domainPtr.initializeNewVectorPtr();

    //     // gather all the domain values
    //     for (vitr = domainObj.Begin(); vitr != domainObj.End(); ++vitr) {
    //         _pushDomainItem(*vitr);
    //     }
    // }

    // assert((mitr = obj.FindMember("range")) != obj.MemberEnd() &&
    //        ((isObject = mitr->value.IsObject()) || (isString = mitr->value.IsString()) ||
    //         (mitr->value.IsArray() && mitr->value.Size())));

    // const rapidjson::Value& rangeObj = mitr->value;

    // if (isObject) {
    //     assert((mitr = rangeObj.FindMember("data")) != rangeObj.MemberEnd() && mitr->value.IsString());
    //     tablePtr = _ctx->getDataTable(mitr->value.GetString());

    //     assert((mitr = rangeObj.FindMember("field")) != rangeObj.MemberEnd() && mitr->value.IsString());
    //     columnPtr = tablePtr->getColumn(mitr->value.GetString());

    //     TDataColumn<RangeType> *rangeColumnPtr = dynamic_cast<TDataColumn<RangeType> *>(columnPtr.get());
    //     // _rangePtr = rangeColumnPtr->getColumnData();
    //     _rangePtr.setVectorPtr(rangeColumnPtr->getColumnData());

    //     if (type == ScaleType::LINEAR) {
    //         std::pair<RangeType, RangeType> minmaxRange = rangeColumnPtr->getExtrema();
    //         // _rangePtr.reset(new std::vector<RangeType>({minmaxRange.first, minmaxRange.second}));
    //         _rangePtr.initializeNewVectorPtr({minmaxRange.first, minmaxRange.second});
    //     }
    // } else if (isString) {
    //     // TODO: Handle actual width/height vals
    //     // _rangePtr.reset(new std::vector<RangeType>({0, 1}));
    //     _rangePtr.initializeNewVectorPtr({0, 1});

    //     // need to put the widht/height range from -1 to 1. Didn't do that
    //     // in the previous line because the compiler was complaining that
    //     // you cannot convert -1 to an unsigned int, tho if range is
    //     // an unsigned int, you should never get here. Doing a subtraction here
    //     // to get a -1 to 1 range without the compilation error.

    //     // (*_rangePtr)[0] -= 1;
    //     _rangePtr.getVectorData()[0] -= 1;
    // } else {
    //     // _rangePtr.reset(new std::vector<RangeType>());
    //     _rangePtr.initializeNewVectorPtr();

    //     // gather all the range values
    //     for (vitr = rangeObj.Begin(); vitr != rangeObj.End(); ++vitr) {
    //         _pushRangeItem(*vitr);
    //     }
    // }
}

template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::_initGLTypes() {
    // _domainType.reset(new TypeGL<DomainType>());
    // _rangeType.reset(new TypeGL<RangeType>());

    _domainType = _domainPtr.getTypeGL();
    _rangeType = _rangePtr.getTypeGL();
}

template <typename DomainType, typename RangeType>
std::string Scale<DomainType, RangeType>::getGLSLCode() {
    // assert(_domainPtr->size() > 0 && _rangePtr->size() > 0);
    assert(_domainPtr.size() > 0 && _rangePtr.size() > 0);

    std::string shaderCode = getShaderCodeFromFile(scaleVertexShaderFilenames[static_cast<int>(type)]);
    std::ostringstream ss;

    boost::replace_first(shaderCode, "<domainType>", _domainType->glslType());

    boost::replace_first(shaderCode, "<rangeType>", _rangeType->glslType());

    // ss << _domainPtr->size();
    ss << _domainPtr.size();
    boost::replace_first(shaderCode, "<numDomains>", ss.str());

    ss.str("");
    ss << _rangePtr.size();
    boost::replace_first(shaderCode, "<numRanges>", ss.str());

    boost::replace_all(shaderCode, "<name>", name);


    if (type == ScaleType::LINEAR) {
        ss.str("");
        ss << _useClamp;
        boost::replace_all(shaderCode, "<useClamp>", ss.str());
    }


    shaderCode += '\n';

    return shaderCode;
}

template <typename DomainType, typename RangeType>
void Scale<DomainType, RangeType>::bindUniformsToRenderer(Shader *activeShader) {
    // activeShader->setUniformAttribute(getDomainGLSLUniformName(), *_domainPtr);
    activeShader->setUniformAttribute(getDomainGLSLUniformName(), _domainPtr.getVectorData());
    // activeShader->setUniformAttribute(getRangeGLSLUniformName(), *_rangePtr);
    activeShader->setUniformAttribute(getRangeGLSLUniformName(), _rangePtr.getVectorData());
}


ScaleShPtr MapD_Renderer::createScale(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx) {
    rapidjson::Value::ConstMemberIterator itr;

    bool isObject;
    assert((itr = obj.FindMember("domain")) != obj.MemberEnd() &&
           ((isObject = itr->value.IsObject()) || (itr->value.IsArray() && itr->value.Size())));

    DataType domainType;

    if (isObject) {
        domainType = getDataTypeFromDataRefJSONObj(itr->value, ctx);
    } else {
        domainType = getDataTypeFromJSONObj(itr->value[0]);
    }


    bool isString;
    assert((itr = obj.FindMember("range")) != obj.MemberEnd() &&
           ((isObject = itr->value.IsObject()) || (isString = itr->value.IsString()) ||
            (itr->value.IsArray() && itr->value.Size())));

    DataType rangeType;
    if (isObject) {
        rangeType = getDataTypeFromDataRefJSONObj(itr->value, ctx);
    } else if (isString) {
        std::string strVal = itr->value.GetString();
        assert(strVal == "width" || strVal == "height");

        // TODO: should we actually use width/height values in the scale?
        // The easy way is just to use the -1 to 1 NDC range, but what
        // if someone uses "height" as the range, but uses that scale
        // for an x coordinate or some other property? Should use the
        // true 0-height in that case.
        rangeType = DataType::FLOAT;
    } else {
        rangeType = getDataTypeFromJSONObj(itr->value[0]);
    }

    switch (domainType) {
        case DataType::UINT:
            switch(rangeType) {
                case DataType::UINT:
                    return ScaleShPtr(new Scale<unsigned int, unsigned int>(obj, ctx));
                case DataType::INT:
                    return ScaleShPtr(new Scale<unsigned int, int>(obj, ctx));
                case DataType::FLOAT:
                    return ScaleShPtr(new Scale<unsigned int, float>(obj, ctx));
                case DataType::DOUBLE:
                    return ScaleShPtr(new Scale<unsigned int, double>(obj, ctx));
                case DataType::COLOR:
                    return ScaleShPtr(new Scale<unsigned int, ColorRGBA>(obj, ctx));
            }
        case DataType::INT:
            switch(rangeType) {
                case DataType::UINT:
                    return ScaleShPtr(new Scale<int, unsigned int>(obj, ctx));
                case DataType::INT:
                    return ScaleShPtr(new Scale<int, int>(obj, ctx));
                case DataType::FLOAT:
                    return ScaleShPtr(new Scale<int, float>(obj, ctx));
                case DataType::DOUBLE:
                    return ScaleShPtr(new Scale<int, double>(obj, ctx));
                case DataType::COLOR:
                    return ScaleShPtr(new Scale<int, ColorRGBA>(obj, ctx));
            }
        case DataType::FLOAT:
            switch(rangeType) {
                case DataType::UINT:
                    return ScaleShPtr(new Scale<float, unsigned int>(obj, ctx));
                case DataType::INT:
                    return ScaleShPtr(new Scale<float, int>(obj, ctx));
                case DataType::FLOAT:
                    return ScaleShPtr(new Scale<float, float>(obj, ctx));
                case DataType::DOUBLE:
                    return ScaleShPtr(new Scale<float, double>(obj, ctx));
                case DataType::COLOR:
                    return ScaleShPtr(new Scale<float, ColorRGBA>(obj, ctx));
            }
        case DataType::DOUBLE:
            switch(rangeType) {
                case DataType::UINT:
                    return ScaleShPtr(new Scale<double, unsigned int>(obj, ctx));
                case DataType::INT:
                    return ScaleShPtr(new Scale<double, int>(obj, ctx));
                case DataType::FLOAT:
                    return ScaleShPtr(new Scale<double, float>(obj, ctx));
                case DataType::DOUBLE:
                    return ScaleShPtr(new Scale<double, double>(obj, ctx));
                case DataType::COLOR:
                    return ScaleShPtr(new Scale<double, ColorRGBA>(obj, ctx));
            }
        case DataType::COLOR:
            switch(rangeType) {
                case DataType::UINT:
                    return ScaleShPtr(new Scale<ColorRGBA, unsigned int>(obj, ctx));
                case DataType::INT:
                    return ScaleShPtr(new Scale<ColorRGBA, int>(obj, ctx));
                case DataType::FLOAT:
                    return ScaleShPtr(new Scale<ColorRGBA, float>(obj, ctx));
                case DataType::DOUBLE:
                    return ScaleShPtr(new Scale<ColorRGBA, double>(obj, ctx));
                case DataType::COLOR:
                    return ScaleShPtr(new Scale<ColorRGBA, ColorRGBA>(obj, ctx));
            }
        default:
            assert(false);
    }
}









































void setRenderPropertyTypeInShaderSrc(const BaseRenderProperty& prop, std::string& shaderSrc) {
    std::ostringstream in_ss, out_ss;

    std::string inname = prop.getInGLSLName();
    std::string intype = prop.getInGLSLType();

    in_ss << "<" << inname << "Type" << ">";
    boost::replace_first(shaderSrc, in_ss.str(), intype);

    std::string outname = prop.getOutGLSLName();
    std::string outtype = prop.getOutGLSLType();

    out_ss << "<" << outname << "Type" << ">";
    boost::replace_first(shaderSrc, out_ss.str(), outtype);
}






void BaseRenderProperty::initializeFromJSONObj(const rapidjson::Value &obj, const DataTableShPtr& dataPtr) {
    if (obj.IsObject()) {
        rapidjson::Value::ConstMemberIterator mitr;

        if ((mitr = obj.FindMember("scale")) != obj.MemberEnd()) {
            // TODO: throw exception instead
            assert(_useScale);

            _initScaleFromJSONObj(mitr->value);
            _verifyScale();
        }


        if ((mitr = obj.FindMember("field")) != obj.MemberEnd()) {
            // TODO: check for an object here
            assert(dataPtr != nullptr);
            assert(mitr->value.IsString());
            // assert(_dataPtr->hasColumn(mitr->value.GetString()));

            initializeFromData(mitr->value.GetString(), dataPtr);
        } else if ((mitr = obj.FindMember("value")) != obj.MemberEnd()) {
            _initValueFromJSONObj(mitr->value, (dataPtr != nullptr ? dataPtr->numRows() : 1));
        } else {
            // TODO: throw error -- need some value source, either
            // by "field" or by "value"
            assert(false);
        }

        _initFromJSONObj(obj);

    } else {
        _initValueFromJSONObj(obj, (dataPtr != nullptr ? dataPtr->numRows() : 1));
    }
}

void BaseRenderProperty::initializeFromData(const std::string& columnName, const DataTableShPtr& dataPtr) {
    // TODO: throw exception instead
    assert(dataPtr != nullptr);

    _vboAttrName = columnName;
    _vboPtr = dataPtr->getColumnDataVBO(columnName);

    if (_scaleConfigPtr == nullptr)
    {
        _initTypeFromVbo();
    }
}

void BaseRenderProperty::_initScaleFromJSONObj(const rapidjson::Value& obj) {
    // assert(obj.IsObject() || obj.IsString());
    assert(obj.IsString());

    assert(_ctx != nullptr && _scaleConfigPtr == nullptr);

    _scaleConfigPtr = _ctx->getScale(obj.GetString());
}

std::string BaseRenderProperty::getInGLSLType() const {
    if (_scaleConfigPtr != nullptr) {
        return _scaleConfigPtr->getDomainType()->glslType();
    }

    // TODO: bug here if no scale is defined, but
    // a data ref is defined.

    assert(_inType != nullptr);
    return _inType->glslType();
}

std::string BaseRenderProperty::getOutGLSLType() const {
    if (_scaleConfigPtr != nullptr) {
        return _scaleConfigPtr->getRangeType()->glslType();
    }

    // TODO: bug here if no scale is defined, but
    // a data ref is defined.

    assert(_outType != nullptr);
    return (_outType->glslType());
}




template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initFromJSONObj(const rapidjson::Value& obj) {
    // this is internally called at the appropriate time from
    // the base class's initialization function, so there's
    // no need to check that obj is valid since that should've
    // already been done.
    rapidjson::Value::ConstMemberIterator mitr;

    if ((mitr = obj.FindMember("mult")) != obj.MemberEnd()) {
        _mult = getNumValFromJSONObj<T>(mitr->value);
    }

    if ((mitr = obj.FindMember("offset")) != obj.MemberEnd()) {
        _offset = getNumValFromJSONObj<T>(mitr->value);
    }
}

template <typename T, int numComponents>
void RenderProperty<T, numComponents>::initializeValue(const T& val, int numItems) {

    // TODO: this is a public function.. should I protect from already existing data?

    _inType.reset(new TypeGL<T, numComponents>());
    _outType.reset(new TypeGL<T, numComponents>());

    std::vector<T> data(numItems, val);

    SequentialBufferLayout *vboLayout = new SequentialBufferLayout();
    vboLayout->addAttribute<T>(_name);

    VertexBuffer::BufferLayoutShPtr vboLayoutPtr;
    vboLayoutPtr.reset(dynamic_cast<BaseBufferLayout *>(vboLayout));

    VertexBuffer *vbo = new VertexBuffer(data, vboLayoutPtr);
    _vboPtr.reset(vbo);
}


template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initValueFromJSONObj(const rapidjson::Value& obj, int numItems) {
    T val = getNumValFromJSONObj<T>(obj);

    initializeValue(val, numItems);
}

template <>
void RenderProperty<ColorRGBA, 1>::_initFromJSONObj(const rapidjson::Value& obj) {
    // TODO: what about offsets / mults for colors?
}

template <>
void RenderProperty<ColorRGBA, 1>::_initValueFromJSONObj(const rapidjson::Value& obj, int numItems) {
    // TODO: throw exception
    assert(obj.IsString());

    ColorRGBA color(obj.GetString());

    // TODO: combine all the different types into a utility file somewhere.
    // i.e. this is already defined in BufferLayout.h, so let's find a
    // good way to consolidate these definitions

    _inType.reset(new TypeGL<float, 4>());
    _outType.reset(new TypeGL<float, 4>());

    std::vector<ColorRGBA> data(numItems, color);

    SequentialBufferLayout *vboLayout = new SequentialBufferLayout();
    vboLayout->addAttribute(_name, BufferAttrType::VEC4F);

    VertexBuffer::BufferLayoutShPtr vboLayoutPtr;
    vboLayoutPtr.reset(dynamic_cast<BaseBufferLayout *>(vboLayout));

    VertexBuffer *vbo = new VertexBuffer(data, vboLayoutPtr);
    // vbo->bufferData();
    _vboPtr.reset(vbo);
}



template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_initTypeFromVbo() {
    assert(_vboPtr != nullptr);

    TypeGLShPtr vboType = _vboPtr->getAttributeTypeGL(_vboAttrName);

    _inType = vboType;
    _outType = vboType;
}

template <>
void RenderProperty<ColorRGBA, 1>::_initTypeFromVbo() {
    assert(_vboPtr != nullptr);

    TypeGLShPtr vboType = _vboPtr->getAttributeTypeGL(_vboAttrName);

    // colors need to be a specific type
    // TODO: Throw an exception instead of an assert
    assert(ColorRGBA::isValidTypeGL(vboType));

    _inType = vboType;
    _outType = vboType;
}




template <typename T, int numComponents>
void RenderProperty<T, numComponents>::_verifyScale() {}

template <>
void RenderProperty<ColorRGBA, 1>::_verifyScale() {
    assert(_scaleConfigPtr != nullptr);

    TypeGLShPtr vboType = _scaleConfigPtr->getRangeType();

    // colors need to be a specific type
    // TODO: Throw an exception instead of an assert
    assert(ColorRGBA::isValidTypeGL(vboType));
}












BaseMark::BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx) : type(geomType), _dataPtr(nullptr), _shaderPtr(nullptr), _vao(0), _ctx(ctx) {}
BaseMark::BaseMark(GeomType geomType, const QueryRendererContextShPtr& ctx, const rapidjson::Value& obj) : BaseMark(geomType, ctx) {
    _initFromJSONObj(obj);
}

BaseMark::~BaseMark() {
    // std::cerr << "IN BaseMark DESTRUCTOR" << std::endl;

    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
    }
}

void BaseMark::_initFromJSONObj(const rapidjson::Value& obj) {
    assert(obj.IsObject());

    rapidjson::Value::ConstMemberIterator mitr;
    assert((mitr = obj.FindMember("properties")) != obj.MemberEnd() &&
           mitr->value.IsObject());

    if ((mitr = obj.FindMember("from")) != obj.MemberEnd()) {
        const rapidjson::Value& fromObj = mitr->value;

        assert(fromObj.IsObject());

        assert((mitr = fromObj.FindMember("data")) != fromObj.MemberEnd() && mitr->value.IsString());

        _dataPtr = _ctx->getDataTable(mitr->value.GetString());
    }
}

void BaseMark::_buildVertexArrayObjectFromProperties(Shader *activeShader) {
    if (_vao) {
        glDeleteVertexArrays(1, &_vao);
        _vao = 0;
    }

    glGenVertexArrays(1, &_vao);
    glBindVertexArray(_vao);

    _initPropertiesForRendering(activeShader);

    // reset so no VAO is bound
    // TODO: Actually look for a currently bound vao at start of function and reset to that?
    // NOTE: No need to unbind at this point since we're going directly into rendering
    // after this function?
    // glBindVertexArray(0);
}

void BaseMark::_bindToRenderer(Shader *activeShader) {
    if (!_vao) {
        _buildVertexArrayObjectFromProperties(activeShader);
    }

    glBindVertexArray(_vao);

    _bindPropertiesToRenderer(activeShader);
}









PointMark::PointMark(
    const rapidjson::Value& obj,
    const QueryRendererContextShPtr& ctx
) : BaseMark(POINTS, ctx, obj),
    x("x", ctx),
    y("y", ctx),
    z("z", ctx),
    size("size", ctx),
    id("id", ctx, false),
    fillColor("fillColor", ctx) {

    _initPropertiesFromJSONObj(obj);
    _initShader();

}

PointMark::~PointMark() {
    // std::cerr << "IN PointGeomConig DESTRUCTOR" << std::endl;
}

void PointMark::_initPropertiesFromJSONObj(const rapidjson::Value& obj) {

    // no need to check for the existence of the 'properties' member as
    // that should've been done by the base class constructor.
    const rapidjson::Value& propObj = obj["properties"];
    rapidjson::Value::ConstMemberIterator mitr;

    assert((mitr = propObj.FindMember("x")) != propObj.MemberEnd() &&
           (mitr->value.IsObject() || mitr->value.IsNumber()));
    x.initializeFromJSONObj(mitr->value, _dataPtr);

    assert((mitr = propObj.FindMember("y")) != propObj.MemberEnd() &&
           (mitr->value.IsObject() || mitr->value.IsNumber()));
    y.initializeFromJSONObj(mitr->value, _dataPtr);

    if ((mitr = propObj.FindMember("z")) != propObj.MemberEnd()) {
        assert(mitr->value.IsObject() || mitr->value.IsNumber());
        z.initializeFromJSONObj(mitr->value, _dataPtr);
    }

    assert((mitr = propObj.FindMember("size")) != propObj.MemberEnd() &&
           (mitr->value.IsObject() || mitr->value.IsNumber()));
    size.initializeFromJSONObj(mitr->value, _dataPtr);

    assert((mitr = propObj.FindMember("fillColor")) != propObj.MemberEnd() &&
           (mitr->value.IsObject() || mitr->value.IsString()));
    fillColor.initializeFromJSONObj(mitr->value, _dataPtr);


    if ((mitr = propObj.FindMember("id")) != propObj.MemberEnd()) {
        assert(mitr->value.IsObject());
        id.initializeFromJSONObj(mitr->value, _dataPtr);
    } else if (_dataPtr != nullptr) {
        id.initializeFromData(DataTable::defaultIdColumnName, _dataPtr);
    } else {
        id.initializeValue(1);  // reaching here "should" guarantee that there's only
                                // 1 row of data
    }
}

void PointMark::_initShader() {
    // TODO: need to determine a build-appropriate way to access
    // shaders
    static const std::string pointVertexShaderFilename = "shaders/pointTemplate.vert";
    static const std::string pointFragmentShaderFilename = "shaders/pointTemplate.frag";

    std::string vertSrc = getShaderCodeFromFile(pointVertexShaderFilename);

    std::vector<BaseRenderProperty *> props = {&x, &y, &size, &id, &fillColor};  // TODO: add z & fillColor

    // update all the types first
    for (auto prop: props) {
        setRenderPropertyTypeInShaderSrc(*prop, vertSrc);
    }

    // now insert any additional functionality
    std::unordered_map<std::string, BaseScale*> visitedScales;
    std::unordered_map<std::string, BaseScale*>::iterator itr;

    std::string funcName;
    std::string propFuncName;
    str_itr_range funcRange;

    for (auto prop: props) {
        ScaleShPtr& scalePtr = prop->getScaleConfig();
        if (scalePtr != nullptr) {
            propFuncName = prop->getGLSLFunc();

            if ((itr = visitedScales.find(scalePtr->name)) == visitedScales.end()) {
                visitedScales.insert(std::make_pair(scalePtr->name, scalePtr.get()));

                std::string scaleCode = scalePtr->getGLSLCode();

                funcRange = getGLSLFunctionBounds(vertSrc, propFuncName);

                // TODO: throw an exception here? Not sure....
                assert(!funcRange.empty());

                boost::replace_range(vertSrc, funcRange, scaleCode);

                // vertSrc.insert(funcInsertPt, scaleCode);
                // funcInsertPt += scaleCode.length();

                funcName = scalePtr->getScaleGLSLFuncName();
            } else {
                funcName = itr->second->getScaleGLSLFuncName();
            }

            // using replace_last to speed up the replace as there should only be 1 occurance
            // and it'll be toward the end of the src string.
            boost::replace_last(vertSrc, prop->getGLSLFunc() + "(" + prop->getName() + ")", funcName + "(" + prop->getName() + ")");
        }
    }

    std::string fragSrc = getShaderCodeFromFile(pointFragmentShaderFilename);


    // CROOT - remote the following includes -- only used for debugging
    // #include <fstream>

    // static int CROOTcnt = 0;
    // CROOTcnt++;
    // if (CROOTcnt == 1) {
    //     std::ofstream shadersrcstream;
    //     shadersrcstream.open("shadersource.vert");
    //     shadersrcstream << vertSrc;
    //     shadersrcstream.close();
    // }


    // now build the shader object
    // TODO: How would we share shaders across different
    // query renderers?
    _shaderPtr.reset(new Shader(vertSrc, fragSrc));
}

void PointMark::_initPropertiesForRendering(Shader *activeShader) {
    assert(x.size() == y.size() && x.size() == size.size() && x.size() == fillColor.size() && x.size() == id.size());

    x.bindToRenderer(activeShader);
    y.bindToRenderer(activeShader);
    // z.bindToRenderer(activeShader);
    size.bindToRenderer(activeShader);
    id.bindToRenderer(activeShader);
    fillColor.bindToRenderer(activeShader);
}

void PointMark::_bindPropertiesToRenderer(Shader *activeShader) {
    std::unordered_map<std::string, BaseScale*> visitedScales;
    std::vector<BaseRenderProperty *> props = {&x, &y, &size, &fillColor};  // TODO: add z & fillColor
    for (auto prop: props) {
        ScaleShPtr& scalePtr = prop->getScaleConfig();
        if (scalePtr != nullptr) {
            if (visitedScales.find(scalePtr->name) == visitedScales.end()) {
                visitedScales.insert(std::make_pair(scalePtr->name, scalePtr.get()));
                scalePtr->bindUniformsToRenderer(activeShader);
            }
        }
    }
}

void PointMark::draw() {
    _shaderPtr->bindToRenderer();

    _bindToRenderer(_shaderPtr.get());

    // TODO: render state stack -- push/pop
    glEnable(GL_PROGRAM_POINT_SIZE);

    // now draw points
    // TODO: What about the possibility of index buffers?
    // Probably don't have to worry about them here since
    // we're specifically looking at a point config that
    // is deemed not to have any index buffers, but we
    // need to have a way to generically draw bound buffers
    // which would be best with a renderer class
    glDrawArrays(GL_POINTS, 0, x.size());

    // unset state
    glDisable(GL_PROGRAM_POINT_SIZE);
}













GeomConfigShPtr MapD_Renderer::createMark(const rapidjson::Value &obj, const QueryRendererContextShPtr& ctx) {
    assert(obj.IsObject());

    rapidjson::Value::ConstMemberIterator itr;
    assert((itr = obj.FindMember("type")) != obj.MemberEnd() &&
           itr->value.IsString());

    BaseMark::GeomType geomType;
    std::string strGeomType(itr->value.GetString());

    if (strGeomType == "points") {
        geomType = BaseMark::POINTS;
        return GeomConfigShPtr(new PointMark(obj, ctx));
    } else {
        assert(false);
    }
}
