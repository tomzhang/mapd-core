#ifndef QUERY_RENDERER_H_
#define QUERY_RENDERER_H_

#include "QueryFramebuffer.h"
#include <unordered_map>
#include <vector>
#include <array>
#include <utility>  // std::pair
#include <string>

#include <gd.h>

namespace MapD_Renderer {

struct DataBuffer {
    GLuint dataType;
    GLuint bufferId;
    // void *data;
};

enum QueryRenderType { MAPD_POINTS=0 };

typedef std::array<float, 4> ColorRGBA;

struct PngData {
    char *pngDataPtr;
    int pngSize;

    PngData(char *pngDataPtr, int pngSize) : pngDataPtr(pngDataPtr), pngSize(pngSize) {}
};

struct ColorConfig {
    std::vector<float> domain;
    std::vector<ColorRGBA> range;
    ColorRGBA defaultColor;
};

struct SizeConfig {
    std::vector<float> domain;
    std::vector<float> range;
    float defaultSize;
};

struct PointRenderConfig {
    SizeConfig sizeConfig;
    ColorConfig colorConfig;
};


struct QueryRenderConfig {
    int width;
    int height;
    float bounds[4];
    QueryRenderType renderType;
    PointRenderConfig renderConfig;
};

typedef std::pair<int, int> UserWidgetIdPair;
typedef std::unordered_map<int, std::unique_ptr<QueryFramebuffer>> WidgetFramebufferMap;
typedef std::unordered_map<int, std::unique_ptr<WidgetFramebufferMap>> FramebufferTable;
typedef std::unordered_map<std::string, DataBuffer> DataTable;

class QueryRenderer {
    public:
        QueryRenderer();
        ~QueryRenderer();

        void render(const DataTable& dataTable, int numRows, const std::string& renderConfigJSON);
        void renderToImage(const DataTable& dataTable, int numRows, const std::string& renderConfigJSON);

        void addFramebuffer(const UserWidgetIdPair& userWidgetIdPair, bool doHitTest=false, bool doDepthTest=false);
        void setActiveFramebufferById(const UserWidgetIdPair& userWidgetIdPair);
        int getActiveUserId() const;
        int getActiveWidgetId() const;

        // get the row id at a specific pixel
        int getIdAt(int x, int y);


        PngData getColorNoisePNG(int width, int height);

    private:
        FramebufferTable _framebufferDict;
        QueryFramebuffer *_activeFramebuffer;
        UserWidgetIdPair _activeFramebufferIds;
};

} // namespace MapD_Renderer

#endif // QUERY_RENDERER_H
