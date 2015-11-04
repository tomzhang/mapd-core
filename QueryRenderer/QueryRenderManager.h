#ifndef QUERY_RENDER_MANAGER_H_
#define QUERY_RENDER_MANAGER_H_

#include "QueryRenderer.h"
#include "Shader.h"
#include "QueryResultVertexBuffer.h"
#include "BufferLayout.h"

#include <GLFW/glfw3.h>
#include <gd.h>

#include <unordered_map>
#include <vector>
#include <array>
#include <utility>  // std::pair
#include <string>
#include <memory>
#include <fstream>
#include <cstdint>
#include <limits>

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

struct QueryDataLayout {
  enum class AttrType { UINT = 0, INT, FLOAT, DOUBLE, INT64 };
  enum class LayoutType { INTERLEAVED = 0, SEQUENTIAL };

  size_t numRows;
  size_t keySz;
  int64_t invalidKey;
  std::vector<std::string> attrNames;
  std::vector<AttrType> attrTypes;
  LayoutType layoutType;

  QueryDataLayout(const size_t numRows,
                  const std::vector<std::string>& attrNames,
                  const std::vector<AttrType>& attrTypes,
                  const size_t keySz = 0,
                  const int64_t invalidKey = std::numeric_limits<int64_t>::max(),
                  const LayoutType layoutType = LayoutType::INTERLEAVED)
      : numRows(numRows),
        keySz(keySz),
        invalidKey(invalidKey),
        attrNames(attrNames),
        attrTypes(attrTypes),
        layoutType(layoutType) {
    // TODO(croot): throw an error instead and quit gracefully?
    CHECK(attrNames.size() == attrTypes.size());
  }

  BufferLayoutShPtr convertToBufferLayout() {
    static const std::string dummyPrefix = "___dummy___";
    int dummyCnt = 0;
    switch (layoutType) {
      case LayoutType::INTERLEAVED: {
        // TODO(croot): make a base interleaved/sequential buffer class
        // perhaps called BaseIntSeqBufferLayout
        // so we don't have to duplicate code here.
        // And support adding the attrnames and types in a constructor of
        // that base class?
        InterleavedBufferLayout* layout = new InterleavedBufferLayout();
        for (size_t i = 0; i < attrNames.size(); ++i) {
          switch (attrTypes[i]) {
            case AttrType::UINT:
              layout->addAttribute(attrNames[i], BufferAttrType::UINT);
              break;
            case AttrType::INT:
              layout->addAttribute(attrNames[i], BufferAttrType::INT);
              break;
            case AttrType::FLOAT:
              layout->addAttribute(attrNames[i], BufferAttrType::FLOAT);
              break;
            case AttrType::DOUBLE:
              layout->addAttribute(attrNames[i], BufferAttrType::DOUBLE);
              break;
            case AttrType::INT64:
              // TODO(croot): support 64-bit ints
              // So for the time being, add a dummy attr for the first
              // 32bits of the attr, and then the real attr for the
              // last 32bits.
              layout->addAttribute(attrNames[i], BufferAttrType::INT);
              layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), BufferAttrType::INT);
              break;
          }
        }
        return BufferLayoutShPtr(layout);
      }
      case LayoutType::SEQUENTIAL: {
        SequentialBufferLayout* layout = new SequentialBufferLayout();
        for (size_t i = 0; i < attrNames.size(); ++i) {
          switch (attrTypes[i]) {
            case AttrType::UINT:
              layout->addAttribute(attrNames[i], BufferAttrType::UINT);
              break;
            case AttrType::INT:
              layout->addAttribute(attrNames[i], BufferAttrType::INT);
              break;
            case AttrType::FLOAT:
              layout->addAttribute(attrNames[i], BufferAttrType::FLOAT);
              break;
            case AttrType::DOUBLE:
              layout->addAttribute(attrNames[i], BufferAttrType::DOUBLE);
              break;
            case AttrType::INT64:
              // TODO(croot): support 64-bit ints
              // So for the time being, add a dummy attr for the first
              // 32bits of the attr, and then the real attr for the
              // last 32bits.
              layout->addAttribute(attrNames[i], BufferAttrType::INT);
              layout->addAttribute(dummyPrefix + std::to_string(dummyCnt++), BufferAttrType::INT);
              break;
          }
        }
        return BufferLayoutShPtr(layout);
      }
      default:
        CHECK(false);
        return nullptr;
    }
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
  void configureRender(const rapidjson::Document& jsonDocument, QueryDataLayout* dataLayoutPtr = nullptr);

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
