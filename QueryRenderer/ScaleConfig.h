#ifndef SCALE_CONFIG_H_
#define SCALE_CONFIG_H_

#include <string>
#include <vector>
#include <iostream>
#include <limits> // std::numeric_limits
#include <memory> // std::unique_ptr
#include "rapidjson/document.h"

namespace MapD_Renderer {

enum class ScaleType { LINEAR = 0, ORDINAL };
enum class DataType { UINT = 0, INT, FLOAT, DOUBLE, COLOR };


class BaseScaleConfig {
    public:
        std::string name;
        ScaleType type;

        DataType domainType;
        DataType rangeType;

        BaseScaleConfig();
        BaseScaleConfig(DataType domain, DataType scale);
        BaseScaleConfig(DataType domain, DataType scale, const rapidjson::Value& obj);

        virtual ~BaseScaleConfig();

        virtual void pushDomainItem(const rapidjson::Value& item) = 0;
        virtual void pushRangeItem(const rapidjson::Value& item) = 0;

    private:
        void _initFromJSONObj(const rapidjson::Value& obj);
};



template <typename DomainType, typename RangeType>
class ScaleConfig : public BaseScaleConfig {
    public:
        std::vector<DomainType> domain;
        std::vector<RangeType> range;

        ScaleConfig();
        ScaleConfig(const rapidjson::Value& obj);

        ~ScaleConfig();

        void pushDomainItem(const rapidjson::Value& obj);
        void pushRangeItem(const rapidjson::Value& obj);

    private:
        void _initFromJSONObj(const rapidjson::Value& obj);
};


// std::unique_ptr<BaseScaleConfig> createScaleConfig(DataType domainType, DataType rangeType);
std::unique_ptr<BaseScaleConfig> createScaleConfig(const rapidjson::Value& obj);



} // MapD_Renderer namespace

#endif // SCALE_CONFIG_H_
