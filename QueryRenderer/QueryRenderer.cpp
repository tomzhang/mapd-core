#include "QueryRenderer.h"
#include "QueryRendererConfig.h"
// #include "QueryRendererConfigJSONHandler.h"
// #include "rapidjson/reader.h"
// #include "rapidjson/error/en.h"
#include "rapidjson/document.h"
#include <iostream>
#include <assert.h>


#include <time.h>
#include <fstream>
#include <map>

using namespace MapD_Renderer;

QueryRenderer::QueryRenderer() : _activeFramebuffer(nullptr) {}
QueryRenderer::~QueryRenderer() {
    std::cerr << "IN QueryRenderer DESTRUCTOR" << std::endl;
}

void QueryRenderer::addFramebuffer(const UserWidgetIdPair& userWidgetIdPair, bool doHitTest, bool doDepthTest) {
    WidgetFramebufferMap *wfMap;
    int userId = userWidgetIdPair.first;
    int widgetId = userWidgetIdPair.second;

    auto userIter = _framebufferDict.find(userId);

    if (userIter == _framebufferDict.end()) {
        wfMap = new WidgetFramebufferMap();
        _framebufferDict[userId] = std::unique_ptr<WidgetFramebufferMap>(wfMap);
    } else {
        wfMap = userIter->second.get();

        if (wfMap->find(widgetId) != wfMap->end()) {
            // a framebuffer already exists! Throw an error.
            // TODO: How should we handle errors?
            std::cerr << "The user id " << userId << " and widget id " << widgetId << " already exists." << std::endl;
            assert(false);
        }
    }

    (*wfMap)[widgetId] = std::unique_ptr<QueryFramebuffer>(new QueryFramebuffer(1, 1, doHitTest, doDepthTest));
}

void QueryRenderer::setActiveFramebufferById(const UserWidgetIdPair& userWidgetIdPair) {
    WidgetFramebufferMap *wfMap;
    int userId = userWidgetIdPair.first;
    int widgetId = userWidgetIdPair.second;

    auto userIter = _framebufferDict.find(userId);

    if (userIter == _framebufferDict.end()) {
        // throw an error
    }

    wfMap = userIter->second.get();
    auto widgetIter = wfMap->find(widgetId);
    if (widgetIter == wfMap->end()) {
        // throw an error
    }

    _activeFramebuffer = widgetIter->second.get();
    _activeFramebufferIds = userWidgetIdPair;
}

int QueryRenderer::getActiveUserId() const {
    if (_activeFramebuffer) {
        return _activeFramebufferIds.first;
    }

    return -1;
}

int QueryRenderer::getActiveWidgetId() const {
    if (_activeFramebuffer) {
        return _activeFramebufferIds.second;
    }

    return -1;
}

void QueryRenderer::render(const DataTable& dataTable, int numRows, const std::string& renderConfigJSON) {
    // rapidjson::Reader reader;
    // QueryRendererConfigJSONHandler handler;
    // rapidjson::StringStream ss(renderConfigJSON.c_str());

    // if (!reader.Parse(ss, handler)) {
    //     // TODO: throw a warning
    //     std::cerr << "Got an error parsing the json config: " << rapidjson::GetParseError_En(reader.GetParseErrorCode()) << std::endl;
    // }

    rapidjson::Document document;
    document.Parse(renderConfigJSON.c_str());

    // QueryRendererConfig config = buildRendererConfigObjFromJSONObject(document);
    QueryRendererConfig config(document);
}

void QueryRenderer::renderToImage(const DataTable& dataTable, int numRows, const std::string& renderConfigJSON) {
    render(dataTable, numRows, renderConfigJSON);
}







int randColor() {
    return rand() % 256;
}

int randAlpha() {
    return rand() % 128;
}



PngData QueryRenderer::getColorNoisePNG(int width, int height) {
    srand(time(NULL));

    // unsigned char* pixels = new unsigned char[width * height * 4];
    int r, g, b, a;

    gdImagePtr im = gdImageCreateTrueColor(width, height);

    std::map<unsigned int, int> colorMap;

    for (int i=0; i<width; ++i) {
        for (int j=0; j<height; ++j) {
            r = randColor();
            g = randColor();
            b = randColor();
            a = 0;

            unsigned int colorId = 16777216 * a + 65536 * b + 256 * g + r;

            if (colorMap.find(colorId) == colorMap.end()) {
                colorMap[colorId] = gdImageColorAllocateAlpha(im, r, g, b, a);
            }

            gdImageSetPixel(im, i, j, colorMap[colorId]);
        }
    }

    int pngSize;
    char* pngPtr = (char*)gdImagePngPtr(im, &pngSize);

    return PngData(pngPtr, pngSize);

    gdImageDestroy(im);
}





