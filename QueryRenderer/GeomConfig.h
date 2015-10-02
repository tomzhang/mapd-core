#ifndef GEOM_CONFIG_H_
#define GEOM_CONFIG_H_

#include "rapidjson/document.h"
#include <memory> // std::unique_ptr
#include "RenderProperty.h"
#include <string>
#include <utility>   // std::pair

namespace MapD_Renderer {

enum GeomType { POINTS = 0 };  // LINES, POLYS



class BaseGeomConfig {
    public:
        GeomType type;

        BaseGeomConfig(GeomType geomType);
        BaseGeomConfig(GeomType geomType, const rapidjson::Value& obj);
        virtual ~BaseGeomConfig();

        // virtual void pushDomainItem(const rapidjson::Value& item) = 0;

        virtual std::pair<std::string, std::string> buildShaderSource() = 0;

    private:
        void _initFromJSONObj(const rapidjson::Value& obj);

    // protected:
    //     typedef std::unique_ptr<Shader> ShaderPtr;
    //     static std::unordered_map<int, ShaderPtr> _shaderMap;
};



class PointGeomConfig : public BaseGeomConfig {
    public:
        PointGeomConfig(const rapidjson::Value& obj);
        ~PointGeomConfig();

        std::pair<std::string, std::string> buildShaderSource();

    private:
        RenderProperty<float> x;
        RenderProperty<float> y;
        RenderProperty<float> z;
        RenderProperty<float> size;
        //RenderProperty<Color> fillColor;

        void _initFromJSONObj(const rapidjson::Value& obj);
        void _initPropertiesFromJSONObj(const rapidjson::Value& propObj);
};


std::unique_ptr<BaseGeomConfig> createGeomConfig(const rapidjson::Value& obj);


} // MapD_Renderer namespace

#endif // GEOM_CONFIG_H_
