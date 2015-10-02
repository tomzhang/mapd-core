#include "GeomConfig.h"
#include <iostream>

using namespace MapD_Renderer;

BaseGeomConfig::BaseGeomConfig(GeomType geomType) : type(geomType) {}
BaseGeomConfig::BaseGeomConfig(GeomType geomType, const rapidjson::Value& obj) : type(geomType) {
    _initFromJSONObj(obj);
}

BaseGeomConfig::~BaseGeomConfig() {
    std::cerr << "IN BaseGeomConfig DESTRUCTOR" << std::endl;
}

void BaseGeomConfig::_initFromJSONObj(const rapidjson::Value& obj) {
    assert(obj.IsObject());

    rapidjson::Value::ConstMemberIterator mitr;
    assert((mitr = obj.FindMember("properties")) != obj.MemberEnd() &&
           mitr->value.IsObject());
}


PointGeomConfig::PointGeomConfig(const rapidjson::Value& obj) : BaseGeomConfig(POINTS, obj), x("x"), y("y"), z("z"), size("size") {
    _initFromJSONObj(obj);
}

PointGeomConfig::~PointGeomConfig() {
    std::cerr << "IN PointGeomConig DESTRUCTOR" << std::endl;
}

void PointGeomConfig::_initFromJSONObj(const rapidjson::Value& obj) {
    // no need to check for the existence of the 'properties' member as
    // that should've been done by the base class constructor.

    const rapidjson::Value& propObj = obj["properties"];
    rapidjson::Value::ConstMemberIterator mitr;

    // assert((mitr = propObj.FindMember("x")) != propObj.MemberEnd() &&
    //        (mitr->value.IsObject() || mitr->value.IsDouble()));

    assert((mitr = propObj.FindMember("x")) != propObj.MemberEnd() &&
           (mitr->value.IsObject() || mitr->value.IsNumber()));
    x.initializeFromJSONObj(mitr->value);

    assert((mitr = propObj.FindMember("y")) != propObj.MemberEnd() &&
           (mitr->value.IsObject() || mitr->value.IsNumber()));
    y.initializeFromJSONObj(mitr->value);

    if ((mitr = propObj.FindMember("z")) != propObj.MemberEnd()) {
        assert(mitr->value.IsObject() || mitr->value.IsNumber());
        z.initializeFromJSONObj(mitr->value);
    }

    assert((mitr = propObj.FindMember("size")) != propObj.MemberEnd() &&
           (mitr->value.IsObject() || mitr->value.IsNumber()));
    size.initializeFromJSONObj(mitr->value);

}

std::unique_ptr<BaseGeomConfig> MapD_Renderer::createGeomConfig(const rapidjson::Value &obj) {
    assert(obj.IsObject());

    rapidjson::Value::ConstMemberIterator itr;
    assert((itr = obj.FindMember("type")) != obj.MemberEnd() &&
           itr->value.IsString());

    GeomType geomType;
    std::string strGeomType(itr->value.GetString());

    if (strGeomType == "points") {
        geomType = POINTS;
        return std::unique_ptr<BaseGeomConfig>(new PointGeomConfig(obj));
    } else {
        assert(false);
    }
}

std::pair<std::string, std::string> PointGeomConfig::buildShaderSource() {

}
