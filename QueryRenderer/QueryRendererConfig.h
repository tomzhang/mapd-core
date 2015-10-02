#ifndef QUERY_RENDERER_CONFIG_H_
#define QUERY_RENDERER_CONFIG_H_

#include <string>
#include <vector>
#include <unordered_map>
#include "ScaleConfig.h"
#include "GeomConfig.h"
#include "rapidjson/document.h"
// #include <utility>  // std::pair
#include <memory>   // std::unique_ptr

namespace MapD_Renderer {

    class QueryRendererConfig {
        public:
            typedef std::unique_ptr<BaseScaleConfig> ScaleConfigPtr;
            typedef std::unique_ptr<BaseGeomConfig> GeomConfigPtr;

            int width;
            int height;
            std::unordered_map<std::string, ScaleConfigPtr> scaleConfigs;
            std::vector<GeomConfigPtr> geomConfigs;

            QueryRendererConfig();
            QueryRendererConfig(const rapidjson::Value& obj);

            ~QueryRendererConfig();

            void set(const rapidjson::Value& obj);

        private:
            void _clear();
            void _initFromJSONObj(const rapidjson::Value& obj);
            // void _buildShaderFromGeomConfig(const GeomConfigPtr& geomConfigPtr);
    };

}; // MapD_Renderer namespace

#endif // QUERY_RENDERER_CONFIG_H_
