#include "ScaleConfig.h"

using namespace MapD_Renderer;

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
                if (val <= std::numeric_limits<float>::max() && val >= std::numeric_limits<float>::min()) {
                    rtn = DataType::FLOAT;
                } else {
                    rtn = DataType::DOUBLE;
                }
            } else {
                assert(false);
            }
            break;
        case rapidjson::kStringType:
            rtn = DataType::COLOR;
            break;
        default:
            assert(false);
    }

    return rtn;
}

BaseScaleConfig::BaseScaleConfig() : name(""), type(ScaleType::LINEAR), domainType(DataType::FLOAT), rangeType(DataType::FLOAT) {}

BaseScaleConfig::BaseScaleConfig(DataType domain, DataType range) : name(""), type(ScaleType::LINEAR), domainType(domain), rangeType(range) {}

BaseScaleConfig::BaseScaleConfig(DataType domain, DataType range, const rapidjson::Value& obj) : BaseScaleConfig(domain, range) {
    _initFromJSONObj(obj);
}

BaseScaleConfig::~BaseScaleConfig() {
    std::cerr << "IN BaseScaleConfig DESTRUCTOR " << name << std::endl;
}

void BaseScaleConfig::_initFromJSONObj(const rapidjson::Value& obj) {
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
    } else if (strScaleType == "ordinal") {
        type = ScaleType::ORDINAL;
    } else {
        assert(false);
    }
}







template <>
ScaleConfig<unsigned int, unsigned int>::ScaleConfig() : BaseScaleConfig(DataType::UINT, DataType::UINT) {}

template <>
ScaleConfig<unsigned int, int>::ScaleConfig() : BaseScaleConfig(DataType::UINT, DataType::INT) {}

template <>
ScaleConfig<unsigned int, float>::ScaleConfig() : BaseScaleConfig(DataType::UINT, DataType::FLOAT) {}

template <>
ScaleConfig<unsigned int, double>::ScaleConfig() : BaseScaleConfig(DataType::UINT, DataType::DOUBLE) {}

// template <>
// ScaleConfig<unsigned int, Color>::ScaleConfig() : BaseScaleConfig(DataType::UINT, DataType::COLOR) {}


template <>
ScaleConfig<int, unsigned int>::ScaleConfig() : BaseScaleConfig(DataType::INT, DataType::UINT) {}

template <>
ScaleConfig<int, int>::ScaleConfig() : BaseScaleConfig(DataType::INT, DataType::INT) {}

template <>
ScaleConfig<int, float>::ScaleConfig() : BaseScaleConfig(DataType::INT, DataType::FLOAT) {}

template <>
ScaleConfig<int, double>::ScaleConfig() : BaseScaleConfig(DataType::INT, DataType::DOUBLE) {}

// template <>
// ScaleConfig<int, Color>::ScaleConfig() : BaseScaleConfig(DataType::INT, DataType::COLOR) {}


template <>
ScaleConfig<float, unsigned int>::ScaleConfig() : BaseScaleConfig(DataType::FLOAT, DataType::UINT) {}

template <>
ScaleConfig<float, int>::ScaleConfig() : BaseScaleConfig(DataType::FLOAT, DataType::INT) {}

template <>
ScaleConfig<float, float>::ScaleConfig() : BaseScaleConfig(DataType::FLOAT, DataType::FLOAT) {}

template <>
ScaleConfig<float, double>::ScaleConfig() : BaseScaleConfig(DataType::FLOAT, DataType::DOUBLE) {}

// template <>
// ScaleConfig<float, Color>::ScaleConfig() : BaseScaleConfig(DataType::FLOAT, DataType::COLOR) {}



template <>
ScaleConfig<double, unsigned int>::ScaleConfig() : BaseScaleConfig(DataType::DOUBLE, DataType::UINT) {}

template <>
ScaleConfig<double, int>::ScaleConfig() : BaseScaleConfig(DataType::DOUBLE, DataType::INT) {}

template <>
ScaleConfig<double, float>::ScaleConfig() : BaseScaleConfig(DataType::DOUBLE, DataType::FLOAT) {}

template <>
ScaleConfig<double, double>::ScaleConfig() : BaseScaleConfig(DataType::DOUBLE, DataType::DOUBLE) {}

// template <>
// ScaleConfig<double, Color>::ScaleConfig() : BaseScaleConfig(DataType::DOUBLE, DataType::COLOR) {}




// template <>
// ScaleConfig<Color, unsigned int>::ScaleConfig() : BaseScaleConfig(DataType::COLOR, DataType::UINT) {}

// template <>
// ScaleConfig<Color, int>::ScaleConfig() : BaseScaleConfig(DataType::COLOR, DataType::INT) {}

// template <>
// ScaleConfig<Color, float>::ScaleConfig() : BaseScaleConfig(DataType::COLOR, DataType::FLOAT) {}

// template <>
// ScaleConfig<Color, double>::ScaleConfig() : BaseScaleConfig(DataType::COLOR, DataType::DOUBLE) {}

// template <>
// ScaleConfig<Color, Color>::ScaleConfig() : BaseScaleConfig(DataType::COLOR, DataType::COLOR) {}












template <>
ScaleConfig<unsigned int, unsigned int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::UINT, DataType::UINT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<unsigned int, int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::UINT, DataType::INT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<unsigned int, float>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::UINT, DataType::FLOAT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<unsigned int, double>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::UINT, DataType::DOUBLE, obj) {
    _initFromJSONObj(obj);
}

// template <>
// ScaleConfig<unsigned int, Color>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::UINT, DataType::COLOR, obj) {
//     _initFromJSONObj(obj);
// }


template <>
ScaleConfig<int, unsigned int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::INT, DataType::UINT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<int, int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::INT, DataType::INT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<int, float>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::INT, DataType::FLOAT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<int, double>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::INT, DataType::DOUBLE, obj) {
    _initFromJSONObj(obj);
}

// template <>
// ScaleConfig<int, Color>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::INT, DataType::COLOR, obj) {
//     _initFromJSONObj(obj);
// }


template <>
ScaleConfig<float, unsigned int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::FLOAT, DataType::UINT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<float, int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::FLOAT, DataType::INT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<float, float>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::FLOAT, DataType::FLOAT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<float, double>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::FLOAT, DataType::DOUBLE, obj) {
    _initFromJSONObj(obj);
}

// template <>
// ScaleConfig<float, Color>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::FLOAT, DataType::COLOR, obj) {
//     _initFromJSONObj(obj);
// }



template <>
ScaleConfig<double, unsigned int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::DOUBLE, DataType::UINT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<double, int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::DOUBLE, DataType::INT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<double, float>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::DOUBLE, DataType::FLOAT, obj) {
    _initFromJSONObj(obj);
}

template <>
ScaleConfig<double, double>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::DOUBLE, DataType::DOUBLE, obj) {
    _initFromJSONObj(obj);
}

// template <>
// ScaleConfig<double, Color>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::DOUBLE, DataType::COLOR, obj) {
//     _initFromJSONObj(obj);
// }




// template <>
// ScaleConfig<Color, unsigned int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::COLOR, DataType::UINT, obj) {
//     _initFromJSONObj(obj);
// }

// template <>
// ScaleConfig<Color, int>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::COLOR, DataType::INT, obj) {
//     _initFromJSONObj(obj);
// }

// template <>
// ScaleConfig<Color, float>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::COLOR, DataType::FLOAT, obj) {
//     _initFromJSONObj(obj);
// }

// template <>
// ScaleConfig<Color, double>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::COLOR, DataType::DOUBLE, obj) {
//     _initFromJSONObj(obj);
// }

// template <>
// ScaleConfig<Color, Color>::ScaleConfig(const rapidjson::Value& obj) : BaseScaleConfig(DataType::COLOR, DataType::COLOR, obj) {
//     _initFromJSONObj(obj);
// }




template <typename DomainType, typename RangeType>
ScaleConfig<DomainType, RangeType>::~ScaleConfig() {
    std::cerr << "IN ScaleConfig DESTRUCTOR " << name << std::endl;
}



// TODO: use template specialization instead of
// a switch inside these functions

template <typename DomainType, typename RangeType>
void ScaleConfig<DomainType, RangeType>::pushDomainItem(const rapidjson::Value& item) {
    DataType type = getDataTypeFromJSONObj(item);
    assert(type == domainType);
    switch(type) {
        case DataType::UINT:
            domain.push_back(item.GetUint());
            break;
        case DataType::INT:
            domain.push_back(item.GetInt());
            break;
        case DataType::FLOAT:
            domain.push_back(static_cast<float>(item.GetDouble()));
            break;
        case DataType::DOUBLE:
            domain.push_back(item.GetDouble());
            break;

        // case DataType::COLOR:
        //     domain.push_back(item.GetString());
        //     break;
        default:
            assert(false);
            break;
    }
}

template <typename DomainType, typename RangeType>
void ScaleConfig<DomainType, RangeType>::pushRangeItem(const rapidjson::Value& item) {
    DataType type = getDataTypeFromJSONObj(item);
    assert(type == rangeType);
    switch(type) {
        case DataType::UINT:
            range.push_back(item.GetUint());
            break;
        case DataType::INT:
            range.push_back(item.GetInt());
            break;
        case DataType::FLOAT:
            range.push_back(static_cast<float>(item.GetDouble()));
            break;
        case DataType::DOUBLE:
            range.push_back(item.GetDouble());
            break;
        // case DataType::COLOR:
        //     range.push_back(item.GetString());
        //     break;
        default:
            assert(false);
            break;
    }
}


template <typename DomainType, typename RangeType>
void ScaleConfig<DomainType, RangeType>::_initFromJSONObj(const rapidjson::Value& obj) {
    rapidjson::Value::ConstMemberIterator mitr;
    rapidjson::Value::ConstValueIterator vitr;

    assert((mitr = obj.FindMember("domain")) != obj.MemberEnd() &&
           mitr->value.IsArray() && mitr->value.Size());
    const rapidjson::Value& domainArray = mitr->value;

    assert((mitr = obj.FindMember("range")) != obj.MemberEnd() &&
           mitr->value.IsArray() && mitr->value.Size());
    const rapidjson::Value& rangeArray = mitr->value;


    // gather all the domain values
    for (vitr = domainArray.Begin(); vitr != domainArray.End(); ++vitr) {
        pushDomainItem(*vitr);
    }

    // gather all the range values
    for (vitr = rangeArray.Begin(); vitr != rangeArray.End(); ++vitr) {
        pushRangeItem(*vitr);
    }
}



std::unique_ptr<BaseScaleConfig> MapD_Renderer::createScaleConfig(const rapidjson::Value& obj) {
    rapidjson::Value::ConstMemberIterator itr;
    assert((itr = obj.FindMember("domain")) != obj.MemberEnd() &&
           itr->value.IsArray() && itr->value.Size());
    DataType domainType = getDataTypeFromJSONObj(itr->value[0]);

    assert((itr = obj.FindMember("range")) != obj.MemberEnd() &&
           itr->value.IsArray() && itr->value.Size());
    DataType rangeType = getDataTypeFromJSONObj(itr->value[0]);

    switch (domainType) {
        case DataType::UINT:
            switch(rangeType) {
                case DataType::UINT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<unsigned int, unsigned int>(obj));
                case DataType::INT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<unsigned int, int>(obj));
                case DataType::FLOAT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<unsigned int, float>(obj));
                case DataType::DOUBLE:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<unsigned int, double>(obj));
                // case DataType::COLOR:
                //     return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<unsigned int, Color>(obj));
            }
        case DataType::INT:
            switch(rangeType) {
                case DataType::UINT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<int, unsigned int>(obj));
                case DataType::INT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<int, int>(obj));
                case DataType::FLOAT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<int, float>(obj));
                case DataType::DOUBLE:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<int, double>(obj));
                // case DataType::COLOR:
                //     return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<int, Color>(obj));
            }
        case DataType::FLOAT:
            switch(rangeType) {
                case DataType::UINT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<float, unsigned int>(obj));
                case DataType::INT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<float, int>(obj));
                case DataType::FLOAT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<float, float>(obj));
                case DataType::DOUBLE:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<float, double>(obj));
                // case DataType::COLOR:
                //     return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<float, Color>(obj));
            }
        case DataType::DOUBLE:
            switch(rangeType) {
                case DataType::UINT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<double, unsigned int>(obj));
                case DataType::INT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<double, int>(obj));
                case DataType::FLOAT:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<double, float>(obj));
                case DataType::DOUBLE:
                    return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<double, double>(obj));
                // case DataType::COLOR:
                //     return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<double, Color>(obj));
            }
        // case DataType::COLOR:
        //     switch(rangeType) {
        //         case DataType::UINT:
        //             return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<Color, unsigned int>(obj));
        //         case DataType::INT:
        //             return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<Color, int>(obj));
        //         case DataType::FLOAT:
        //             return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<Color, float>(obj));
        //         case DataType::DOUBLE:
        //             return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<Color, double>(obj));
        //         case DataType::COLOR:
        //             return std::unique_ptr<BaseScaleConfig>(new ScaleConfig<Color, Color>(obj));
        //     }
        default:
            assert(false);
    }
}

