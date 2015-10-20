#ifndef QUERY_RENDER_MANAGER_H_
#define QUERY_RENDER_MANAGER_H_

#include "QueryRenderer.h"
#include "Shader.h"
#include "VertexBuffer.h"

#include <unordered_map>
#include <vector>
#include <array>
#include <utility>  // std::pair
#include <string>
#include <memory>
#include <GLFW/glfw3.h>
#include <fstream>

#include <gd.h>

namespace MapD_Renderer {

struct PngData {
    char *pngDataPtr;
    int pngSize;

    PngData(char *pngDataPtr, int pngSize) : pngDataPtr(pngDataPtr), pngSize(pngSize) {}

    void writeToFile(const std::string& filename) {
        std::ofstream pngFile(filename, std::ios::binary);
        pngFile.write(pngDataPtr, pngSize);
        pngFile.close();
    }
};

typedef std::pair<int, int> UserWidgetPair;
typedef std::unordered_map<int, QueryRendererUqPtr> WidgetRendererMap;
typedef std::unordered_map<int, std::unique_ptr<WidgetRendererMap>> RendererTable;

class QueryRenderManager {
    public:
        explicit QueryRenderManager(int queryResultBufferSize = 500000, bool debugMode=false);
        ~QueryRenderManager();

        bool inDebugMode() const;

        bool hasUser(int userId) const;
        bool hasUserWidget(int userId, int widgetId) const;
        bool hasUserWidget(const UserWidgetPair& userWidgetPair) const;

        void addUserWidget(int userId, int widgetId, const std::string& configJSON, bool doHitTest=false, bool doDepthTest=false);
        void addUserWidget(const UserWidgetPair& userWidgetPair, const std::string& configJSON, bool doHitTest=false, bool doDepthTest=false);

        void setJSONConfigForUserWidget(int userId, int widgetId, const std::string& configJSON);
        void setJSONConfigForUserWidget(const UserWidgetPair& userWidgetPair, const std::string& configJSON);

        void setUserWidgetWidthHeight(int userId, int widgetId, int width, int height);
        void setUserWidgetWidthHeight(const UserWidgetPair& userWidgetPair, int width, int height);

        void renderUserWidget(int userId, int widgetId) const;
        void renderUserWidget(const UserWidgetPair& userWidgetPair) const;

        PngData renderToPng(int userId, int widgetId) const;
        PngData renderToPng(const UserWidgetPair& userWidgetPair) const;

        // int getActiveUser() const;
        // int getActiveWidgetId() const;

        // get the id at a specific pixel
        unsigned int getIdAt(int userId, int widgetId, int x, int y) const;
        unsigned int getIdAt(const UserWidgetPair& userWidgetPair, int x, int y) const;


        PngData getColorNoisePNG(int width, int height);


        // CROOT - the following is a debug function. Remove when ready for deployment
        GLFWwindow * getWindow() {
            // return _windowPtr.get();
            return _windowPtr;
        }

    private:
        bool _debugMode;
        RendererTable _rendererDict;

        mutable QueryRenderer *_activeRenderer;
        mutable UserWidgetPair _activeUserWidget;

        // std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> _windowPtr;
        GLFWwindow *_windowPtr;
        VertexBufferUqPtr _queryResultBuffer;

        void _initGLFW();
        void _initQueryResultBuffer(int queryResultBufferSize);
        void _setActiveUserWidget(int userId, int widgetId) const;
        QueryRenderer* _getRendererForUserWidget(int userId, int widgetId) const;
};

} // namespace MapD_Renderer

#endif // QUERY_RENDER_MANAGER_H_
