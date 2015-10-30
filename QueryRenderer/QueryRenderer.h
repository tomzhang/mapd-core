#ifndef QUERY_RENDERER_H_
#define QUERY_RENDERER_H_

#include "QueryResultVertexBuffer.h"
#include "QueryFramebuffer.h"
#include "QueryRendererObjects.h"
#include "QueryFramebuffer.h"
// #include "QueryRenderManager.h"
#include "DataTable.h"
#include "BufferLayout.h"

#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>  // std::unique_ptr
#include <cstdint>
#include <limits>

#include "rapidjson/document.h"
// #include <utility>  // std::pair

namespace MapD_Renderer {

class QueryRendererContext;
class BaseMark;
class BaseScale;

class QueryRenderer {
 public:
  explicit QueryRenderer(const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false,
                         GLFWwindow* win = nullptr);

  explicit QueryRenderer(const rapidjson::Document& jsonDocument,
                         const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false,
                         GLFWwindow* win = nullptr);

  explicit QueryRenderer(const std::string& configJSON,
                         const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                         bool doHitTest = false,
                         bool doDepthTest = false,
                         GLFWwindow* win = nullptr);

  ~QueryRenderer();

  int getWidth();
  int getHeight();
  void setWidthHeight(int width, int height, GLFWwindow* win = nullptr);

  const QueryFramebufferUqPtr& getFramebuffer();

  void setJSONConfig(const std::string& configJSON, GLFWwindow* win = nullptr);
  void setJSONDocument(const rapidjson::Document& jsonDocument, GLFWwindow* win = nullptr);

  void updateQueryResultBufferPostQuery(const BufferLayoutShPtr& layoutPtr,
                                        const int numRows,
                                        const int64_t invalid_key);

  void render();

  unsigned int getIdAt(int x, int y);

 private:
  std::shared_ptr<QueryRendererContext> _ctx;
  QueryFramebufferUqPtr _framebufferPtr;

  void _clear();
  void _initFromJSON(const rapidjson::Value& obj, GLFWwindow* win = nullptr);
  void _initFromJSON(const std::string& configJSON, GLFWwindow* win = nullptr);
  void _initFramebuffer(int width, int height);
};

typedef std::unique_ptr<QueryRenderer> QueryRendererUqPtr;
typedef std::shared_ptr<QueryRenderer> QueryRendererShPtr;

class QueryRendererContext {
 public:
  typedef std::shared_ptr<BaseScale> ScaleShPtr;

  explicit QueryRendererContext(const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                                bool doHitTest = false,
                                bool doDepthTest = false)
      : _queryResultVBOPtr(queryResultVBOPtr),
        _width(0),
        _height(0),
        _doHitTest(doHitTest),
        _doDepthTest(doDepthTest),
        _invalidKey(std::numeric_limits<int64_t>::max()) {}
  explicit QueryRendererContext(const QueryResultVertexBufferShPtr& queryResultVBOPtr,
                                int width,
                                int height,
                                bool doHitTest = false,
                                bool doDepthTest = false)
      : _queryResultVBOPtr(queryResultVBOPtr),
        _width(width),
        _height(height),
        _doHitTest(doHitTest),
        _doDepthTest(doDepthTest),
        _invalidKey(std::numeric_limits<int64_t>::max()) {}
  ~QueryRendererContext() { _clear(); }

  int getWidth() { return _width; }

  int getHeight() { return _height; }

  bool hasDataTable(const std::string& tableName) const;
  DataVBOShPtr getDataTable(const std::string& tableName) const;

  bool hasScale(const std::string& scaleConfigName) const;
  ScaleShPtr getScale(const std::string& scaleConfigName) const;

  QueryResultVertexBufferShPtr getQueryResultVertexBuffer() const { return _queryResultVBOPtr; }

  bool doHitTest() { return _doHitTest; }
  bool doDepthTest() { return _doDepthTest; }

  int64_t getInvalidKey() { return _invalidKey; }

  // bool hasMark(const std::string& geomConfigName) const {
  //     return (_geomConfigMap.find(geomConfigName) != _geomConfigMap.end());
  // }

  // GeomConfigShPtr getMark(const std::string& geomConfigName) {
  //     GeomConfigShPtr rtn(nullptr);

  //     auto itr = _geomConfigMap.find(geomConfigName);
  //     if (itr != _geomConfigMap.end()) {
  //         rtn = itr->second;
  //     }

  //     return rtn;
  // }

  friend class QueryRenderer;

 private:
  typedef std::unordered_map<std::string, ScaleShPtr> ScaleConfigMap;

  typedef std::shared_ptr<BaseMark> GeomConfigShPtr;

  typedef std::vector<GeomConfigShPtr> GeomConfigVector;
  typedef std::unordered_map<std::string, DataVBOShPtr> DataTableMap;

  DataTableMap _dataTableMap;
  ScaleConfigMap _scaleConfigMap;
  GeomConfigVector _geomConfigs;
  QueryResultVertexBufferShPtr _queryResultVBOPtr;
  BufferLayoutShPtr _queryResultBufferLayout;

  int _width;
  int _height;
  bool _doHitTest;
  bool _doDepthTest;
  int64_t _invalidKey;

  void _clear() {
    _width = 0;
    _height = 0;
    _dataTableMap.clear();
    _scaleConfigMap.clear();
    _geomConfigs.clear();
  }
};

typedef std::unique_ptr<QueryRendererContext> QueryRendererContextUqPtr;
typedef std::shared_ptr<QueryRendererContext> QueryRendererContextShPtr;

DataVBOShPtr createDataTable(const rapidjson::Value& obj, const QueryRendererContextShPtr& ctx);

};  // MapD_Renderer namespace

#endif  // QUERY_RENDERER_H_
