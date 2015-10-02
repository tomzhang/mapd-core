#include "QueryRendererConfig.h"

using namespace MapD_Renderer;

QueryRendererConfig::QueryRendererConfig() : width(0), height(0) {}
QueryRendererConfig::QueryRendererConfig(const rapidjson::Value& obj) : width(0), height(0) {
    _initFromJSONObj(obj);
}

QueryRendererConfig::~QueryRendererConfig() {
    std::cerr << "IN QueryRendererConfig DESTRUCTOR: " << scaleConfigs.size() << std::endl;
    // scaleConfigs.clear();
}

void QueryRendererConfig::set(const rapidjson::Value& obj) {
    _clear();
    _initFromJSONObj(obj);
}


void QueryRendererConfig::_clear() {
    width = 0;
    height = 0;
    scaleConfigs.clear();
}

void QueryRendererConfig::_initFromJSONObj(const rapidjson::Value& obj) {
    assert(obj.IsObject());

    rapidjson::Value::ConstMemberIterator mitr;
    rapidjson::Value::ConstValueIterator vitr;

    assert((mitr = obj.FindMember("width")) != obj.MemberEnd() &&
           mitr->value.IsInt());
    width = mitr->value.GetInt();

    assert((mitr = obj.FindMember("height")) != obj.MemberEnd() &&
           mitr->value.IsInt());
    height = mitr->value.GetInt();

    mitr = obj.FindMember("scales");
    if (mitr != obj.MemberEnd()) {
        assert(mitr->value.IsArray());

        for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
            std::unique_ptr<BaseScaleConfig> scaleConfig = createScaleConfig(*vitr);
            scaleConfigs.insert(std::make_pair(scaleConfig->name, std::move(scaleConfig)));
        }
    }

    mitr = obj.FindMember("marks");
    if (mitr != obj.MemberEnd()) {
        assert(mitr->value.IsArray());

        for (vitr = mitr->value.Begin(); vitr != mitr->value.End(); ++vitr) {
            geomConfigs.push_back(createGeomConfig(*vitr));


        }
    }
}

// void QueryRendererConfig::_buildShaderFromGeomConfig(const GeomConfigPtr& geomConfigPtr) {
// }
