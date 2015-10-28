#ifndef QUERY_RENDER_MANAGER_H_
#define QUERY_RENDER_MANAGER_H_

#include "QueryRenderer.h"
#include "Shader.h"
#include "QueryResultVertexBuffer.h"

#include <GLFW/glfw3.h>
#include <gd.h>

#include <unordered_map>
#include <vector>
#include <array>
#include <utility>  // std::pair
#include <string>
#include <memory>
#include <fstream>

#include "rapidjson/document.h"

namespace MapD_Renderer {

struct PngData {
  char* pngDataPtr;
  int pngSize;

  PngData(char* pngDataPtr, int pngSize) : pngDataPtr(pngDataPtr), pngSize(pngSize) {}

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
  explicit QueryRenderManager(unsigned int queryResultBufferSize = 500000, bool debugMode = false);
  ~QueryRenderManager();

  CudaHandle getCudaHandle();

  bool inDebugMode() const;

  bool hasUser(int userId) const;
  bool hasUserWidget(int userId, int widgetId) const;
  bool hasUserWidget(const UserWidgetPair& userWidgetPair) const;

  void addUserWidget(int userId, int widgetId, bool doHitTest = false, bool doDepthTest = false);
  void addUserWidget(const UserWidgetPair& userWidgetPair, bool doHitTest = false, bool doDepthTest = false);

  void setActiveUserWidget(int userId, int widgetId);
  void setActiveUserWidget(const UserWidgetPair& userWidgetPair);

  void setWidthHeight(int width, int height);

  // TODO(croot): add result buffer layout object
  void configureRender(const rapidjson::Document& jsonDocument);

  void render() const;
  PngData renderToPng() const;

  // get the id at a specific pixel
  unsigned int getIdAt(int x, int y) const;

  PngData getColorNoisePNG(int width, int height);

  // CROOT - the following is a debug function. Remove when ready for deployment
  GLFWwindow* getWindow() {
    // return _windowPtr.get();
    return _windowPtr;
  }

 private:
  bool _debugMode;
  RendererTable _rendererDict;

  mutable QueryRenderer* _activeRenderer;
  mutable UserWidgetPair _activeUserWidget;

  // std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> _windowPtr;
  GLFWwindow* _windowPtr;
  QueryResultVertexBufferShPtr _queryResultVBOPtr;
  unsigned int _queryResultBufferSize;

  void _initGLFW();
  void _initQueryResultBuffer();
  void _setActiveUserWidget(int userId, int widgetId) const;
  QueryRenderer* _getRendererForUserWidget(int userId, int widgetId) const;
};

}  // namespace MapD_Renderer

#endif  // QUERY_RENDER_MANAGER_H_
